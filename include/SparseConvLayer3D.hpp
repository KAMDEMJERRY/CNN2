#pragma once
#include "SparseTensor.hpp"
#include "Optimizer.hpp"
#include <Eigen/Dense>
#include <random>
#include <string>
#include <stdexcept>
#include <cmath>
#include <vector>

// =============================================================================
// SparseConvLayer3D — Convolution 3D sparse (SubManifold Sparse Convolution)
//
// Principe SubManifold :
//   Une position de sortie (b, od, oh, ow) est active si et seulement si
//   la position d'entrée correspondante (b, od, oh, ow) était active.
//   → La sparsité est PRÉSERVÉE à travers les couches, pas amplifiée.
//
// Algorithme im2col sparse :
//   Au lieu de boucler sur TOUTES les positions de sortie (coûteux),
//   on boucle uniquement sur les N voxels actifs en entrée.
//   Pour chaque voxel actif (b, d, h, w), on construit sa ligne im2col
//   en cherchant chaque voisin (b, d+kd-p, h+kh-p, w+kw-p) dans la
//   hash map de l'entrée. Si absent → contribution nulle (zero-padding).
//
// Différences avec ConvLayer3D dense :
//   - Entrée/sortie : SparseTensor au lieu de Tensor
//   - im2col : boucle sur nnz, lookup O(1) par voisin
//   - col2im : accumulation sparse → grad_input SparseTensor
//   - Pas de stride > 1 en mode SubManifold (stride=1 obligatoire)
//     Pour strider, utiliser SparseConvLayer3D en mode "standard sparse"
//     avec stride > 1 (voir constructeur : submanifold=false)
//
// Poids : même layout que ConvLayer3D — (out_ch, in_ch, Kd, Kh, Kw)
// Normalisation des gradients : identique à ConvLayer3D — division par B
// =============================================================================
class SparseConvLayer3D {
public:

    // ── Constructeur ──────────────────────────────────────────────────────────

    SparseConvLayer3D(int in_channels,  int out_channels,
                      int kernel_d,     int kernel_h,   int kernel_w,
                      int stride_d = 1, int stride_h = 1, int stride_w = 1,
                      int pad_d    = 0, int pad_h    = 0, int pad_w    = 0,
                      bool submanifold = true)
        : in_ch_(in_channels),   out_ch_(out_channels),
          kd_(kernel_d),   kh_(kernel_h),   kw_(kernel_w),
          sd_(stride_d),   sh_(stride_h),   sw_(stride_w),
          pd_(pad_d),      ph_(pad_h),      pw_(pad_w),
          submanifold_(submanifold)
    {
        if (submanifold_ && (sd_ != 1 || sh_ != 1 || sw_ != 1))
            throw std::runtime_error(
                "[SparseConvLayer3D] Mode SubManifold impose stride=1. "
                "Utilisez submanifold=false pour stride > 1.");

        // Initialisation des poids et biais
        const int K = kd_ * kh_ * kw_;
        weights_    = Eigen::MatrixXf(out_ch_, in_ch_ * K);
        bias_       = Eigen::VectorXf::Zero(out_ch_);
        grad_w_     = Eigen::MatrixXf::Zero(out_ch_, in_ch_ * K);
        grad_b_     = Eigen::VectorXf::Zero(out_ch_);
        isTrainable = true;
        initializeWeights("he");
    }

    ~SparseConvLayer3D() = default;

    SparseConvLayer3D(const SparseConvLayer3D&)            = delete;
    SparseConvLayer3D& operator=(const SparseConvLayer3D&) = delete;

    bool isTrainable = true;
    std::string getName() const { return "SparseConvLayer3D"; }

    // ── Initialisation des poids ──────────────────────────────────────────────

    // Identique à ConvLayer3D : He pour ReLU, Xavier pour tanh/sigmoid
    void initializeWeights(const std::string& method = "he") {
        const int fan_in = in_ch_ * kd_ * kh_ * kw_;
        const float scale = (method == "he")
            ? std::sqrt(2.0f / fan_in)
            : std::sqrt(1.0f / fan_in);

        std::mt19937 gen{std::random_device{}()};
        std::normal_distribution<float> dist(0.0f, scale);

        for (int i = 0; i < weights_.rows(); ++i)
            for (int j = 0; j < weights_.cols(); ++j)
                weights_(i, j) = dist(gen);

        bias_.setZero();
        grad_w_.setZero();
        grad_b_.setZero();
    }

    // ── Forward pass ──────────────────────────────────────────────────────────
    //
    // Entrée  : SparseTensor (B, C_in,  D, H, W)
    // Sortie  : SparseTensor (B, C_out, D_out, H_out, W_out)
    //
    // Mode SubManifold : D_out=D, H_out=H, W_out=W  (sparsité préservée)
    // Mode Standard    : D_out, H_out, W_out selon stride/pad (sparsité croît)
    SparseTensor forward(const SparseTensor& input) {
        // Validation
        if (input.num_channels != in_ch_)
            throw std::runtime_error(
                "[SparseConvLayer3D::forward] in_channels mismatch: attendu "
                + std::to_string(in_ch_) + ", reçu "
                + std::to_string(input.num_channels));

        // Cache pour le backward
        input_cache_ = input;
        input.buildLookup(); // s'assure que la lookup table est à jour

        const int B  = input.batch_size;
        const int D  = input.spatial_d;
        const int H  = input.spatial_h;
        const int W  = input.spatial_w;
        const int N  = input.nnz();
        const int K  = kd_ * kh_ * kw_;

        // Dimensions de sortie
        const int oD = submanifold_ ? D : (D + 2*pd_ - kd_) / sd_ + 1;
        const int oH = submanifold_ ? H : (H + 2*ph_ - kh_) / sh_ + 1;
        const int oW = submanifold_ ? W : (W + 2*pw_ - kw_) / sw_ + 1;

        // ── Étape 1 : Déterminer les positions de sortie actives ──────────────
        //
        // SubManifold : une sortie est active ⟺ l'entrée correspondante est active
        // → on copie directement les coords d'entrée
        //
        // Standard sparse : une sortie est active si au moins un voisin en entrée
        // est actif → on collecte toutes les positions de sortie couvertes
        std::vector<std::array<int,4>> out_coords_vec;

        if (submanifold_) {
            // Exactement les mêmes positions que l'entrée
            out_coords_vec.reserve(N);
            for (int i = 0; i < N; ++i)
                out_coords_vec.push_back({input.coords(i,0), input.coords(i,1),
                                          input.coords(i,2), input.coords(i,3)});
        } else {
            // Collecte des positions de sortie uniques couvertes par des voxels actifs
            // On utilise un set via unordered_map pour dédupliquer
            SparseTensor tmp_out;
            tmp_out.batch_size  = B;
            tmp_out.spatial_d   = oD;
            tmp_out.spatial_h   = oH;
            tmp_out.spatial_w   = oW;
            std::unordered_map<uint64_t, bool> seen;
            seen.reserve(N * K);

            for (int i = 0; i < N; ++i) {
                const int b = input.coords(i,0);
                const int d = input.coords(i,1);
                const int h = input.coords(i,2);
                const int w = input.coords(i,3);
                // Positions de sortie couvertes par ce voxel d'entrée
                for (int kd = 0; kd < kd_; ++kd)
                for (int kh = 0; kh < kh_; ++kh)
                for (int kw = 0; kw < kw_; ++kw) {
                    // Position de sortie od telle que id = od*sd - pd + kd
                    // → od = (id + pd - kd) / sd  (entier si divisible)
                    const int num_d = d + pd_ - kd;
                    const int num_h = h + ph_ - kh;
                    const int num_w = w + pw_ - kw;
                    if (num_d < 0 || num_d % sd_ != 0) continue;
                    if (num_h < 0 || num_h % sh_ != 0) continue;
                    if (num_w < 0 || num_w % sw_ != 0) continue;
                    const int od = num_d / sd_;
                    const int oh = num_h / sh_;
                    const int ow = num_w / sw_;
                    if (od < 0 || od >= oD || oh < 0 || oh >= oH || ow < 0 || ow >= oW) continue;
                    const uint64_t key = tmp_out.encode(b, od, oh, ow);
                    if (!seen.count(key)) {
                        seen[key] = true;
                        out_coords_vec.push_back({b, od, oh, ow});
                    }
                }
            }
        }

        const int N_out = static_cast<int>(out_coords_vec.size());

        // ── Étape 2 : Construction im2col sparse ──────────────────────────────
        //
        // col_matrix : (in_ch * Kd * Kh * Kw, N_out)
        // Chaque colonne j correspond au voxel de sortie j.
        // La cellule (row, j) = feature du voisin d'entrée correspondant
        // (0 si le voisin est absent = zero padding implicite).
        //
        // row = ic * (Kd*Kh*Kw) + kd * (Kh*Kw) + kh * Kw + kw

        Eigen::MatrixXf col_matrix = Eigen::MatrixXf::Zero(in_ch_ * K, N_out);

        for (int j = 0; j < N_out; ++j) {
            const int b  = out_coords_vec[j][0];
            const int od = out_coords_vec[j][1];
            const int oh = out_coords_vec[j][2];
            const int ow = out_coords_vec[j][3];

            for (int kd = 0; kd < kd_; ++kd)
            for (int kh = 0; kh < kh_; ++kh)
            for (int kw = 0; kw < kw_; ++kw) {
                // Position d'entrée correspondante
                const int id = od * sd_ - pd_ + kd;
                const int ih = oh * sh_ - ph_ + kh;
                const int iw = ow * sw_ - pw_ + kw;

                // Hors bornes → contribution nulle (déjà 0)
                if (id < 0 || id >= D || ih < 0 || ih >= H || iw < 0 || iw >= W)
                    continue;

                // Cherche le voxel d'entrée (b, id, ih, iw) dans la hash map
                const int idx = input.find(b, id, ih, iw);
                if (idx < 0) continue; // voxel inactif → contribution nulle

                // Copie les features de ce voxel dans la colonne im2col
                const int row_base = (kd * kh_ + kh) * kw_ + kw;
                for (int ic = 0; ic < in_ch_; ++ic) {
                    col_matrix(ic * K + row_base, j) = input.features(idx, ic);
                }
            }
        }

        // ── Étape 3 : GEMM ────────────────────────────────────────────────────
        // weights_ : (out_ch, in_ch * K)
        // col_matrix : (in_ch * K, N_out)
        // out_mat : (out_ch, N_out)
        Eigen::MatrixXf out_mat = weights_ * col_matrix;
        out_mat.colwise() += bias_;

        // ── Étape 4 : Assemblage du SparseTensor de sortie ────────────────────
        SparseTensor output;
        output.batch_size   = B;
        output.num_channels = out_ch_;
        output.spatial_d    = oD;
        output.spatial_h    = oH;
        output.spatial_w    = oW;
        output.coords.resize(N_out, 4);
        output.features.resize(N_out, out_ch_);

        for (int j = 0; j < N_out; ++j) {
            output.coords(j, 0) = out_coords_vec[j][0];
            output.coords(j, 1) = out_coords_vec[j][1];
            output.coords(j, 2) = out_coords_vec[j][2];
            output.coords(j, 3) = out_coords_vec[j][3];
            output.features.row(j) = out_mat.col(j).transpose();
        }

        // Mise en cache pour le backward
        col_cache_ = col_matrix;
        out_coords_cache_ = out_coords_vec;

        return output;
    }

    // ── Backward pass ─────────────────────────────────────────────────────────
    //
    // grad_output : SparseTensor de même structure que la sortie du forward
    //
    // Calcule :
    //   dL/dW = grad_out_mat × col_cache^T / B
    //   dL/db = sum(grad_out_mat, axis=colonnes) / (B * N_out)
    //   dL/dX : via col2im sparse (accumulation += dans SparseTensor)
    SparseTensor backward(const SparseTensor& grad_output) {
        const int N_out = grad_output.nnz();
        const int B     = input_cache_.batch_size;
        const int K     = kd_ * kh_ * kw_;

        if (N_out != static_cast<int>(out_coords_cache_.size()))
            throw std::runtime_error(
                "[SparseConvLayer3D::backward] N_out mismatch avec le forward cache");

        // grad_out_mat : (out_ch, N_out)
        Eigen::MatrixXf grad_out_mat(out_ch_, N_out);
        for (int j = 0; j < N_out; ++j)
            grad_out_mat.col(j) = grad_output.features.row(j).transpose();

        // ── dL/dW ─────────────────────────────────────────────────────────────
        // Même convention que ConvLayer3D : division par B (batch size)
        grad_w_ = (grad_out_mat * col_cache_.transpose()) / static_cast<float>(B);

        // ── dL/db ─────────────────────────────────────────────────────────────
        grad_b_ = grad_out_mat.rowwise().sum()
                / static_cast<float>(B * std::max(N_out, 1));

        // ── dL/dX : col2im sparse ─────────────────────────────────────────────
        // dX_col = W^T × grad_out_mat → (in_ch * K, N_out)
        const Eigen::MatrixXf dX_col = weights_.transpose() * grad_out_mat;

        // Accumulateur de gradient sur les voxels d'entrée
        // On réutilise les coordonnées de input_cache_
        const int N_in = input_cache_.nnz();
        Eigen::MatrixXf grad_in_features = Eigen::MatrixXf::Zero(N_in, in_ch_);

        // Index inverse : coord d'entrée → indice dans input_cache_
        input_cache_.buildLookup();

        for (int j = 0; j < N_out; ++j) {
            const int b  = out_coords_cache_[j][0];
            const int od = out_coords_cache_[j][1];
            const int oh = out_coords_cache_[j][2];
            const int ow = out_coords_cache_[j][3];

            for (int kd = 0; kd < kd_; ++kd)
            for (int kh = 0; kh < kh_; ++kh)
            for (int kw = 0; kw < kw_; ++kw) {
                const int id = od * sd_ - pd_ + kd;
                const int ih = oh * sh_ - ph_ + kh;
                const int iw = ow * sw_ - pw_ + kw;

                if (id < 0 || id >= input_cache_.spatial_d) continue;
                if (ih < 0 || ih >= input_cache_.spatial_h) continue;
                if (iw < 0 || iw >= input_cache_.spatial_w) continue;

                const int idx = input_cache_.find(b, id, ih, iw);
                if (idx < 0) continue;

                const int row_base = (kd * kh_ + kh) * kw_ + kw;
                for (int ic = 0; ic < in_ch_; ++ic) {
                    grad_in_features(idx, ic) +=
                        dX_col(ic * K + row_base, j);
                }
            }
        }

        // Construction du SparseTensor gradient d'entrée
        SparseTensor grad_input;
        grad_input.batch_size   = input_cache_.batch_size;
        grad_input.num_channels = in_ch_;
        grad_input.spatial_d    = input_cache_.spatial_d;
        grad_input.spatial_h    = input_cache_.spatial_h;
        grad_input.spatial_w    = input_cache_.spatial_w;
        grad_input.coords       = input_cache_.coords;
        grad_input.features     = grad_in_features;

        return grad_input;
    }

    // ── Mise à jour des poids via l'optimiseur ────────────────────────────────
    // Compatible avec Adam et SGD — même interface que ConvLayer3D
    void updateParams(Optimizer& optimizer) {
        optimizer.updateWeights(weights_, grad_w_);
        optimizer.updateBias   (bias_,    grad_b_);
        grad_w_.setZero();
        grad_b_.setZero();
    }

    // ── Accesseurs ────────────────────────────────────────────────────────────

    Eigen::MatrixXf& getWeights()         { return weights_; }
    Eigen::MatrixXf& getWeightGradients() { return grad_w_;  }
    Eigen::VectorXf& getBias()            { return bias_;    }
    Eigen::VectorXf& getBiasGradients()   { return grad_b_;  }

    void setWeights(const Eigen::MatrixXf& w) {
        if (w.rows() != out_ch_ || w.cols() != in_ch_ * kd_ * kh_ * kw_)
            throw std::runtime_error("[SparseConvLayer3D::setWeights] dimensions incorrectes");
        weights_ = w;
    }

    void setBias(const Eigen::VectorXf& b) {
        if (b.size() != out_ch_)
            throw std::runtime_error("[SparseConvLayer3D::setBias] taille incorrecte");
        bias_ = b;
    }

    // Dimensions de sortie (utile pour CNNBuilder sparse)
    struct OutDims { int d, h, w; };
    OutDims outputDims(int in_d, int in_h, int in_w) const {
        if (submanifold_) return {in_d, in_h, in_w};
        return {
            (in_d + 2*pd_ - kd_) / sd_ + 1,
            (in_h + 2*ph_ - kh_) / sh_ + 1,
            (in_w + 2*pw_ - kw_) / sw_ + 1
        };
    }

    bool isSubmanifold() const { return submanifold_; }

    int inChannels()  const { return in_ch_;  }
    int outChannels() const { return out_ch_; }
    int numParams()   const { return out_ch_ * in_ch_ * kd_ * kh_ * kw_ + out_ch_; }

private:

    // ── Hyperparamètres ───────────────────────────────────────────────────────
    int in_ch_, out_ch_;
    int kd_, kh_, kw_;
    int sd_, sh_, sw_;
    int pd_, ph_, pw_;
    bool submanifold_;

    // ── Paramètres apprenables ────────────────────────────────────────────────
    // Layout : (out_ch, in_ch * Kd * Kh * Kw)  — même ordre que ConvLayer3D
    Eigen::MatrixXf weights_;
    Eigen::VectorXf bias_;

    // ── Gradients ─────────────────────────────────────────────────────────────
    Eigen::MatrixXf grad_w_;
    Eigen::VectorXf grad_b_;

    // ── Cache backward ────────────────────────────────────────────────────────
    SparseTensor                     input_cache_;
    Eigen::MatrixXf                  col_cache_;
    std::vector<std::array<int,4>>   out_coords_cache_;
};
