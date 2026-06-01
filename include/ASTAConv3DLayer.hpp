#pragma once
#include "Layer.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <string>
#include <stdexcept>
#include <vector>
#include <algorithm>

// =============================================================================
// ASTAConv3DLayer — Attention Spatio-Temporelle Axiale pour Conv3D
// =============================================================================
//
// ASTA = Axial Spatial-Temporal Attention (Conv3D)
//
// Cette couche combine deux mécanismes d'attention complémentaires appliqués
// à des features 3D médicaux (B, C, D, H, W) :
//
// ── 1. Attention de canal (Channel Attention — SE-like) ──────────────────────
//
//   Identifie QUELS canaux sont les plus informatifs.
//
//   a) Squeeze : Global Average Pool sur (D, H, W) → vecteur (B, C)
//   b) Excitation :
//        FC1 : C → C/r (reduction_ratio r, défaut 4) + ReLU
//        FC2 : C/r → C + Sigmoid
//   c) Recalibrage : multiplier chaque canal par son score d'attention
//
// ── 2. Attention spatiale 3D (Spatial Attention) ─────────────────────────────
//
//   Identifie OÙ dans le volume se trouvent les régions pertinentes.
//
//   a) Pool de canal : avg + max sur la dimension C → 2 cartes (B, 1, D, H, W)
//   b) Concaténation → (B, 2, D, H, W)
//   c) Convolution 3D 1×7×7 axiale → 1 canal + Sigmoid (carte d'attention)
//       Note : kernel asymétrique 1×7×7 → axial (pas de réduction D)
//   d) Multiplication point-à-point avec les features
//
// ── 3. Connexion résiduelle + LayerNorm ─────────────────────────────────────
//
//   output = LayerNorm(x_channel_scaled ⊙ x_spatial_attn + input)
//
// ── Référence ────────────────────────────────────────────────────────────────
//
//   Inspiré de CBAM (Woo et al., 2018) adapté aux volumes 3D médicaux,
//   avec attention axiale (Huang et al., 2019) pour limiter la complexité.
//   Complexité : O(C²/r + C×D×H×W) — nettement inférieure à l'attention
//   par tokens O(N² × C) avec N = D×H×W.
//
// ── Usage ─────────────────────────────────────────────────────────────────────
//
//   model.addLayer(std::make_shared<ASTAConv3DLayer>(
//       64,    // channels
//       4,     // reduction_ratio pour Channel Attention
//       true,  // use_residual
//       true   // use_layer_norm
//   ));
//
// ── Paramètres apprenables ────────────────────────────────────────────────────
//
//   W1_ca  : (C, C/r)  — FC1 channel attention
//   W2_ca  : (C/r, C)  — FC2 channel attention
//   W_sa   : (1, 2, 1, 7, 7) — conv axiale spatial attention
//   b_sa   : (1,)      — biais conv spatiale
//   gamma  : (C,)      — scale LayerNorm
//   beta   : (C,)      — shift LayerNorm
//
// =============================================================================

class ASTAConv3DLayer : public Layer {
public:

    // ── Constructeur ──────────────────────────────────────────────────────────
    ASTAConv3DLayer(int  channels,
                    int  reduction_ratio = 4,
                    bool use_residual    = true,
                    bool use_norm        = true)
        : C_(channels),
          r_(reduction_ratio),
          use_residual_(use_residual),
          use_norm_(use_norm)
    {
        if (C_ <= 0)
            throw std::invalid_argument("[ASTAConv3D] channels doit être > 0");
        if (r_ <= 0)
            throw std::invalid_argument("[ASTAConv3D] reduction_ratio doit être > 0");

        Cr_ = std::max(1, C_ / r_);  // dimension réduite (au moins 1)

        // ── Channel Attention ─────────────────────────────────────────────────
        W1_ca_ = Eigen::MatrixXf(Cr_, C_);   // FC1 : C → Cr (matrice transposée)
        W2_ca_ = Eigen::MatrixXf(C_,  Cr_);  // FC2 : Cr → C

        dW1_ca_ = Eigen::MatrixXf::Zero(Cr_, C_);
        dW2_ca_ = Eigen::MatrixXf::Zero(C_,  Cr_);

        b1_ca_ = Eigen::VectorXf::Zero(Cr_);
        b2_ca_ = Eigen::VectorXf::Zero(C_);

        db1_ca_ = Eigen::VectorXf::Zero(Cr_);
        db2_ca_ = Eigen::VectorXf::Zero(C_);

        // ── Spatial Attention (conv axiale 1×7×7) ────────────────────────────
        // Poids : (1, 2, 1, 7, 7) → matrice (49, 2) pour la convolution
        // (1 filtre de sortie, 2 canaux d'entrée avg/max, kernel 1×7×7)
        kH_ = 7; kW_ = 7;  // kernel spatial
        W_sa_  = Eigen::MatrixXf(kH_ * kW_, 2);  // (49, 2)
        dW_sa_ = Eigen::MatrixXf::Zero(kH_ * kW_, 2);
        b_sa_  = 0.0f;
        db_sa_ = 0.0f;

        // ── LayerNorm ─────────────────────────────────────────────────────────
        ln_gamma_ = Eigen::VectorXf::Ones(C_);
        ln_beta_  = Eigen::VectorXf::Zero(C_);
        dln_gamma_ = Eigen::VectorXf::Zero(C_);
        dln_beta_  = Eigen::VectorXf::Zero(C_);

        isTrainable = true;
        initWeights();
    }

    ~ASTAConv3DLayer() override = default;

    ASTAConv3DLayer(const ASTAConv3DLayer&)            = delete;
    ASTAConv3DLayer& operator=(const ASTAConv3DLayer&) = delete;

    // ── Initialisation Xavier ─────────────────────────────────────────────────
    void initWeights() {
        std::mt19937 gen{std::random_device{}()};
        auto xavier = [&](Eigen::MatrixXf& M, int fan_in, int fan_out) {
            const float s = std::sqrt(2.0f / (fan_in + fan_out));
            std::normal_distribution<float> d(0.f, s);
            for (int r = 0; r < M.rows(); ++r)
                for (int c = 0; c < M.cols(); ++c)
                    M(r, c) = d(gen);
        };
        xavier(W1_ca_, C_,         Cr_);
        xavier(W2_ca_, Cr_,        C_);
        xavier(W_sa_,  2 * kH_ * kW_, 1);
        // Biais nuls
        b1_ca_.setZero(); b2_ca_.setZero();
        // LayerNorm : gamma=1, beta=0
        ln_gamma_.setOnes(); ln_beta_.setZero();
    }

    // =========================================================================
    // FORWARD
    // =========================================================================
    Tensor forward(const Tensor& input) override {
        if (input.ndim() != 5)
            throw std::runtime_error(
                "[ASTAConv3D::forward] Attend Tensor 5D (B,C,D,H,W)");
        if (input.dim(1) != C_)
            throw std::runtime_error(
                "[ASTAConv3D::forward] Mauvais nombre de canaux : "
                + std::to_string(input.dim(1)) + " vs " + std::to_string(C_));

        input_cache_ = input;

        const int B = input.dim(0);
        const int D = input.dim(2);
        const int H = input.dim(3);
        const int W = input.dim(4);

        // ── Étape 1 : Channel Attention ───────────────────────────────────────
        //
        // squeeze : (B, C) — moyenne globale sur (D, H, W)
        Eigen::MatrixXf gap(B, C_);   // global average pool
        gap.setZero();
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C_; ++c)
                for (int d = 0; d < D; ++d)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w)
                            gap(b, c) += input(b, c, d, h, w);
        gap /= static_cast<float>(D * H * W);

        gap_cache_ = gap;

        // FC1 : (B, Cr) = gap * W1^T + b1
        Eigen::MatrixXf z1 = gap * W1_ca_.transpose();  // (B, Cr)
        z1.rowwise() += b1_ca_.transpose();
        // ReLU
        z1_cache_ = z1;
        z1 = z1.cwiseMax(0.0f);
        relu1_cache_ = z1;

        // FC2 : (B, C) = relu1 * W2^T + b2
        Eigen::MatrixXf z2 = z1 * W2_ca_.transpose();  // (B, C)
        z2.rowwise() += b2_ca_.transpose();
        z2_cache_ = z2;

        // Sigmoid → poids canal
        Eigen::MatrixXf ca = sigmoid(z2);  // (B, C)
        ca_cache_ = ca;

        // Recalibrage canal : multiplier features par poids
        Tensor x_ca(B, C_, D, H, W);
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C_; ++c)
                for (int d = 0; d < D; ++d)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w)
                            x_ca(b, c, d, h, w) = input(b, c, d, h, w) * ca(b, c);

        x_ca_cache_ = x_ca;

        // ── Étape 2 : Spatial Attention (axiale 1×7×7) ───────────────────────
        //
        // Projections de canal : avg et max sur la dim C → (B, 1, D, H, W)
        // Concaténation → (B, 2, D, H, W)
        // Convolution axiale 1×7×7 → (B, 1, D, H, W) → sigmoid

        // avg_map et max_map : (B, D, H, W)
        avg_map_.assign(B * D * H * W, 0.0f);
        max_map_.assign(B * D * H * W, -1e38f);

        for (int b = 0; b < B; ++b)
            for (int d = 0; d < D; ++d)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w) {
                        const int idx = ((b * D + d) * H + h) * W + w;
                        float sum = 0.f, mx = -1e38f;
                        for (int c = 0; c < C_; ++c) {
                            float v = x_ca(b, c, d, h, w);
                            sum += v;
                            if (v > mx) mx = v;
                        }
                        avg_map_[idx] = sum / static_cast<float>(C_);
                        max_map_[idx] = mx;
                    }

        // Convolution axiale 1×7×7 avec padding 0×3×3 sur (avg, max) → score(b,d,h,w)
        const int pH = kH_ / 2;  // padding H = 3
        const int pW = kW_ / 2;  // padding W = 3

        sa_scores_.assign(B * D * H * W, 0.0f);  // avant sigmoid

        for (int b = 0; b < B; ++b)
            for (int d = 0; d < D; ++d)       // axial : pas de kernel sur D
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w) {
                        float val = b_sa_;
                        for (int kh = 0; kh < kH_; ++kh)
                            for (int kw = 0; kw < kW_; ++kw) {
                                const int ih = h + kh - pH;
                                const int iw = w + kw - pW;
                                const int kidx = kh * kW_ + kw;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    const int fidx = ((b * D + d) * H + ih) * W + iw;
                                    val += W_sa_(kidx, 0) * avg_map_[fidx]
                                         + W_sa_(kidx, 1) * max_map_[fidx];
                                }
                            }
                        sa_scores_[((b * D + d) * H + h) * W + w] = val;
                    }

        // Sigmoid → carte d'attention spatiale
        sa_attn_.resize(B * D * H * W);
        for (int i = 0; i < static_cast<int>(sa_scores_.size()); ++i)
            sa_attn_[i] = sigmoidf(sa_scores_[i]);

        // Appliquer la carte spatiale sur x_ca
        Tensor output(B, C_, D, H, W);
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C_; ++c)
                for (int d = 0; d < D; ++d)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w) {
                            const float a = sa_attn_[((b * D + d) * H + h) * W + w];
                            output(b, c, d, h, w) = x_ca(b, c, d, h, w) * a;
                        }

        // ── Étape 3 : Résiduelle + LayerNorm ─────────────────────────────────
        if (use_residual_)
            for (int i = 0; i < output.size(); ++i)
                output[i] += input[i];

        if (use_norm_)
            output = layerNorm(output, B, D, H, W);

        return output;
    }

    // =========================================================================
    // BACKWARD
    // =========================================================================
    Tensor backward(const Tensor& grad_output) override {
        const int B = input_cache_.dim(0);
        const int D = input_cache_.dim(2);
        const int H = input_cache_.dim(3);
        const int W = input_cache_.dim(4);
        const float inv_B = 1.f / static_cast<float>(B);

        // ── Backward LayerNorm ────────────────────────────────────────────────
        Tensor grad = grad_output;
        if (use_norm_)
            grad = layerNormBackward(grad, B, D, H, W);

        // Gradient vers l'entrée (résiduelle)
        Tensor grad_residual(grad.shape());
        grad_residual.setZero();
        if (use_residual_)
            grad_residual = grad;

        // ── Backward Spatial Attention ────────────────────────────────────────
        //
        // output(b,c,d,h,w) = x_ca(b,c,d,h,w) * sa_attn_(b,d,h,w)
        // → d_x_ca = grad * sa_attn_
        // → d_sa_attn = sum_c(grad * x_ca)

        std::vector<float> d_sa_attn(B * D * H * W, 0.f);
        Tensor d_x_ca(B, C_, D, H, W);
        d_x_ca.setZero();

        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C_; ++c)
                for (int d = 0; d < D; ++d)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w) {
                            const int idx = ((b * D + d) * H + h) * W + w;
                            const float a = sa_attn_[idx];
                            const float g = grad(b, c, d, h, w);
                            d_x_ca(b, c, d, h, w) = g * a;
                            d_sa_attn[idx] += g * x_ca_cache_(b, c, d, h, w);
                        }

        // Backward sigmoid
        std::vector<float> d_sa_scores(B * D * H * W, 0.f);
        for (int i = 0; i < static_cast<int>(d_sa_scores.size()); ++i) {
            const float s = sa_attn_[i];
            d_sa_scores[i] = d_sa_attn[i] * s * (1.f - s);
        }

        // Backward convolution axiale 1×7×7 → d_avg_map, d_max_map, dW_sa_, db_sa_
        const int pH = kH_ / 2;
        const int pW = kW_ / 2;

        std::vector<float> d_avg_map(B * D * H * W, 0.f);
        std::vector<float> d_max_map(B * D * H * W, 0.f);

        for (int b = 0; b < B; ++b)
            for (int d = 0; d < D; ++d)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w) {
                        const float ds = d_sa_scores[((b * D + d) * H + h) * W + w];
                        db_sa_ += ds * inv_B;
                        for (int kh = 0; kh < kH_; ++kh)
                            for (int kw = 0; kw < kW_; ++kw) {
                                const int ih = h + kh - pH;
                                const int iw = w + kw - pW;
                                const int kidx = kh * kW_ + kw;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    const int fidx = ((b * D + d) * H + ih) * W + iw;
                                    dW_sa_(kidx, 0) += ds * avg_map_[fidx] * inv_B;
                                    dW_sa_(kidx, 1) += ds * max_map_[fidx] * inv_B;
                                    d_avg_map[fidx] += ds * W_sa_(kidx, 0);
                                    d_max_map[fidx] += ds * W_sa_(kidx, 1);
                                }
                            }
                    }

        // Backward avg_map / max_map → d_x_ca
        // avg_map = mean_c(x_ca) → grad vers chaque canal : d_avg/C
        // max_map = max_c(x_ca)  → grad vers le canal max_c seulement
        for (int b = 0; b < B; ++b)
            for (int d = 0; d < D; ++d)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w) {
                        const int idx = ((b * D + d) * H + h) * W + w;
                        const float da = d_avg_map[idx] / static_cast<float>(C_);
                        // avg → distribué uniformément
                        for (int c = 0; c < C_; ++c)
                            d_x_ca(b, c, d, h, w) += da;
                        // max → seulement au canal qui avait le max
                        float mx = -1e38f;
                        int cmax = 0;
                        for (int c = 0; c < C_; ++c) {
                            if (x_ca_cache_(b, c, d, h, w) > mx) {
                                mx = x_ca_cache_(b, c, d, h, w);
                                cmax = c;
                            }
                        }
                        d_x_ca(b, cmax, d, h, w) += d_max_map[idx];
                    }

        // ── Backward Channel Attention ────────────────────────────────────────
        //
        // x_ca = input * ca   →  d_input_ca = d_x_ca * ca
        //                         d_ca = sum_{d,h,w}(d_x_ca * input) / (D*H*W)

        Tensor grad_input(B, C_, D, H, W);
        grad_input.setZero();

        Eigen::MatrixXf d_ca(B, C_);
        d_ca.setZero();

        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C_; ++c)
                for (int d = 0; d < D; ++d)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w) {
                            float g = d_x_ca(b, c, d, h, w);
                            grad_input(b, c, d, h, w) += g * ca_cache_(b, c);
                            d_ca(b, c) += g * input_cache_(b, c, d, h, w);
                        }

        // Backward sigmoid CA
        Eigen::MatrixXf d_z2(B, C_);
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C_; ++c) {
                const float s = ca_cache_(b, c);
                d_z2(b, c) = d_ca(b, c) * s * (1.f - s);
            }

        // Backward FC2 : z2 = relu1 * W2^T + b2
        db2_ca_ += (d_z2.colwise().sum() * inv_B).transpose();
        dW2_ca_ += (d_z2.transpose() * relu1_cache_) * inv_B;
        Eigen::MatrixXf d_relu1 = d_z2 * W2_ca_;  // (B, Cr)

        // Backward ReLU1
        Eigen::MatrixXf d_z1(B, Cr_);
        for (int b = 0; b < B; ++b)
            for (int cr = 0; cr < Cr_; ++cr)
                d_z1(b, cr) = (z1_cache_(b, cr) > 0.f) ? d_relu1(b, cr) : 0.f;

        // Backward FC1 : z1 = gap * W1^T + b1
        db1_ca_ += (d_z1.colwise().sum() * inv_B).transpose();
        dW1_ca_ += (d_z1.transpose() * gap_cache_) * inv_B;
        Eigen::MatrixXf d_gap = d_z1 * W1_ca_;  // (B, C)

        // Backward GAP : distribuez uniformément sur (D,H,W)
        const float inv_DHW = 1.f / static_cast<float>(D * H * W);
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C_; ++c)
                for (int d = 0; d < D; ++d)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w)
                            grad_input(b, c, d, h, w) += d_gap(b, c) * inv_DHW;

        // ── Résiduelle ────────────────────────────────────────────────────────
        if (use_residual_)
            for (int i = 0; i < grad_input.size(); ++i)
                grad_input[i] += grad_residual[i];

        return grad_input;
    }

    // ── Mise à jour des poids ─────────────────────────────────────────────────
    void updateParams(Optimizer& optimizer) override {
        optimizer.updateWeights(W1_ca_, dW1_ca_);
        optimizer.updateWeights(W2_ca_, dW2_ca_);
        optimizer.updateWeights(W_sa_,  dW_sa_);
        optimizer.updateBias(b1_ca_, db1_ca_);
        optimizer.updateBias(b2_ca_, db2_ca_);
        optimizer.updateBias(ln_gamma_, dln_gamma_);
        optimizer.updateBias(ln_beta_,  dln_beta_);

        // Scalaire b_sa_ mis à jour manuellement (pas de wrapper Eigen)
        // On utilise updateBias via un VectorXf temporaire
        Eigen::VectorXf bsa_v(1), dbsa_v(1);
        bsa_v(0) = b_sa_; dbsa_v(0) = db_sa_;
        optimizer.updateBias(bsa_v, dbsa_v);
        b_sa_ = bsa_v(0);

        // Remise à zéro des gradients
        dW1_ca_.setZero(); dW2_ca_.setZero(); dW_sa_.setZero();
        db1_ca_.setZero(); db2_ca_.setZero();
        dln_gamma_.setZero(); dln_beta_.setZero();
        db_sa_ = 0.f;
    }

    // ── Sérialisation ─────────────────────────────────────────────────────────
    void saveParameters(boost::archive::binary_oarchive& ar) const override {
        ar << W1_ca_ << W2_ca_ << W_sa_;
        ar << b1_ca_ << b2_ca_ << b_sa_;
        ar << ln_gamma_ << ln_beta_;
    }

    void loadParameters(boost::archive::binary_iarchive& ar) override {
        ar >> W1_ca_ >> W2_ca_ >> W_sa_;
        ar >> b1_ca_ >> b2_ca_ >> b_sa_;
        ar >> ln_gamma_ >> ln_beta_;
    }

    // ── Informations ─────────────────────────────────────────────────────────
    std::string getName() const override {
        return "ASTAConv3D(C=" + std::to_string(C_)
             + " r="  + std::to_string(r_)
             + " k1x" + std::to_string(kH_) + "x" + std::to_string(kW_)
             + (use_residual_ ? " res" : "")
             + (use_norm_     ? " ln"  : "") + ")";
    }

    int numParams() const override {
        // CA : W1(Cr×C) + W2(C×Cr) + b1(Cr) + b2(C)
        // SA : W_sa(kH×kW, 2) + b_sa(1)
        // LN : gamma(C) + beta(C)
        return Cr_ * C_ + C_ * Cr_ + Cr_ + C_
             + kH_ * kW_ * 2 + 1
             + (use_norm_ ? 2 * C_ : 0);
    }

    // Accesseurs (tests)
    Eigen::MatrixXf& getW1CA() { return W1_ca_; }
    Eigen::MatrixXf& getW2CA() { return W2_ca_; }
    Eigen::MatrixXf& getWSA()  { return W_sa_;  }

private:

    // ── Hyperparamètres ───────────────────────────────────────────────────────
    int  C_, r_, Cr_;     // canaux, ratio réduction, dim réduite
    int  kH_, kW_;        // kernel spatial attention (7×7)
    bool use_residual_, use_norm_;

    // ── Paramètres Channel Attention ─────────────────────────────────────────
    Eigen::MatrixXf W1_ca_, W2_ca_;          // FC1 (Cr×C), FC2 (C×Cr)
    Eigen::VectorXf b1_ca_, b2_ca_;          // biais FC1 (Cr), FC2 (C)
    Eigen::MatrixXf dW1_ca_, dW2_ca_;
    Eigen::VectorXf db1_ca_, db2_ca_;

    // ── Paramètres Spatial Attention ─────────────────────────────────────────
    Eigen::MatrixXf W_sa_;    // (kH×kW, 2) — filtre conv axiale
    float           b_sa_;
    Eigen::MatrixXf dW_sa_;
    float           db_sa_ = 0.f;

    // ── LayerNorm ─────────────────────────────────────────────────────────────
    Eigen::VectorXf ln_gamma_, ln_beta_;
    Eigen::VectorXf dln_gamma_, dln_beta_;

    // ── Cache backward ────────────────────────────────────────────────────────
    Tensor          input_cache_;
    Tensor          x_ca_cache_;       // après channel attention
    Eigen::MatrixXf gap_cache_;        // (B, C) — global avg pool
    Eigen::MatrixXf z1_cache_;         // avant ReLU1 (B, Cr)
    Eigen::MatrixXf relu1_cache_;      // après ReLU1 (B, Cr)
    Eigen::MatrixXf z2_cache_;         // avant sigmoid FC2 (B, C)
    Eigen::MatrixXf ca_cache_;         // (B, C) — poids canal après sigmoid
    std::vector<float> avg_map_;       // (B×D×H×W)
    std::vector<float> max_map_;       // (B×D×H×W)
    std::vector<float> sa_scores_;    // avant sigmoid spatial
    std::vector<float> sa_attn_;      // après sigmoid spatial

    // ── Cache LayerNorm ───────────────────────────────────────────────────────
    Eigen::MatrixXf ln_x_norm_;
    Eigen::VectorXf ln_mean_, ln_var_;
    int             ln_N_ = 0;

    // ── Utilitaires ───────────────────────────────────────────────────────────
    static float sigmoidf(float x) { return 1.f / (1.f + std::exp(-x)); }

    static Eigen::MatrixXf sigmoid(const Eigen::MatrixXf& M) {
        return (1.f + (-M.array()).exp()).inverse().matrix();
    }

    static int ceilDiv(int a, int b) { return (a + b - 1) / b; }

    // ── LayerNorm forward ─────────────────────────────────────────────────────
    Tensor layerNorm(const Tensor& t, int B, int D, int H, int W) {
        const float eps = 1e-5f;
        const int   N   = B * D * H * W;
        Eigen::MatrixXf X(N, C_);
        int idx = 0;
        for (int b=0;b<B;++b) for (int d=0;d<D;++d)
        for (int h=0;h<H;++h) for (int w=0;w<W;++w) {
            for (int c=0;c<C_;++c) X(idx,c) = t(b,c,d,h,w);
            ++idx;
        }
        ln_mean_ = X.rowwise().mean();
        Eigen::MatrixXf Xc = X.colwise() - ln_mean_;
        ln_var_  = Xc.array().square().rowwise().mean();
        ln_x_norm_.resize(N, C_);
        for (int i=0;i<N;++i) {
            float si = 1.f / std::sqrt(ln_var_[i] + eps);
            for (int c=0;c<C_;++c) ln_x_norm_(i,c) = Xc(i,c)*si;
        }
        ln_N_ = N;
        Eigen::MatrixXf Y = (ln_x_norm_.array().rowwise()
                             * ln_gamma_.transpose().array())
                            .rowwise() + ln_beta_.transpose().array();
        Tensor out(B, C_, D, H, W);
        idx = 0;
        for (int b=0;b<B;++b) for (int d=0;d<D;++d)
        for (int h=0;h<H;++h) for (int w=0;w<W;++w) {
            for (int c=0;c<C_;++c) out(b,c,d,h,w) = Y(idx,c);
            ++idx;
        }
        return out;
    }

    // ── LayerNorm backward ────────────────────────────────────────────────────
    Tensor layerNormBackward(const Tensor& grad, int B, int D, int H, int W) {
        const float eps = 1e-5f;
        Eigen::MatrixXf dY(ln_N_, C_);
        int idx = 0;
        for (int b=0;b<B;++b) for (int d=0;d<D;++d)
        for (int h=0;h<H;++h) for (int w=0;w<W;++w) {
            for (int c=0;c<C_;++c) dY(idx,c) = grad(b,c,d,h,w);
            ++idx;
        }
        const float inv_B = 1.f / static_cast<float>(B);
        dln_gamma_ += ((dY.array() * ln_x_norm_.array()).colwise().sum()
                       * inv_B).matrix().transpose();
        dln_beta_  += (dY.colwise().sum() * inv_B).transpose();

        Eigen::MatrixXf dX(ln_N_, C_);
        for (int i=0;i<ln_N_;++i) {
            float si = 1.f / std::sqrt(ln_var_[i] + eps);
            Eigen::RowVectorXf dyi  = dY.row(i);
            Eigen::RowVectorXf xni  = ln_x_norm_.row(i);
            float sum_dY    = (dyi.array() * ln_gamma_.transpose().array()).sum();
            float sum_dY_xn = (dyi.array() * ln_gamma_.transpose().array()
                               * xni.array()).sum();
            dX.row(i) = si / static_cast<float>(C_)
                * ((dyi.array() * ln_gamma_.transpose().array()
                    * static_cast<float>(C_))
                   - sum_dY - xni.array() * sum_dY_xn).matrix();
        }
        Tensor out(B, C_, D, H, W);
        idx = 0;
        for (int b=0;b<B;++b) for (int d=0;d<D;++d)
        for (int h=0;h<H;++h) for (int w=0;w<W;++w) {
            for (int c=0;c<C_;++c) out(b,c,d,h,w) = dX(idx,c);
            ++idx;
        }
        return out;
    }
};
