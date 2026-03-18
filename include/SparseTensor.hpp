#pragma once
#include "Tensor.hpp"
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <algorithm>

// =============================================================================
// SparseTensor — représentation creuse d'un volume 3D pour CNN sparse
//
// Stockage :
//   coords   : Eigen::MatrixXi (N, 4)  — chaque ligne = [batch, d, h, w]
//   features : Eigen::MatrixXf (N, C)  — chaque ligne = vecteur de features
//
// Invariant : coords et features ont le même nombre de lignes N.
//   N = nombre de voxels actifs dans le batch entier.
//
// Relation avec Tensor dense (B, C, D, H, W) :
//   - from_dense() : Tensor → SparseTensor  (seuillage)
//   - to_dense()   : SparseTensor → Tensor  (reconstruction)
//
// Clé de hash pour la lookup table :
//   key(b, d, h, w) = b*(D*H*W) + d*(H*W) + h*W + w  (uint64_t)
// =============================================================================
class SparseTensor {
public:

    // ── Stockage principal ────────────────────────────────────────────────────

    // coords(i, :) = [batch, d, h, w] du i-ème voxel actif
    Eigen::MatrixXi coords;   // (N, 4)  int32

    // features(i, :) = vecteur de canaux du i-ème voxel actif
    Eigen::MatrixXf features; // (N, C)  float32

    // Dimensions spatiales d'origine (nécessaires pour to_dense et les couches)
    int batch_size    = 0;
    int num_channels  = 0;
    int spatial_d     = 0;
    int spatial_h     = 0;
    int spatial_w     = 0;

    // ── Table de lookup : coord → indice de ligne dans coords/features ────────
    // Construite à la demande via buildLookup()
    // key = encode(b,d,h,w) → index i dans [0, N[
    mutable std::unordered_map<uint64_t, int> lookup;
    mutable bool lookup_valid = false;

    // ── Constructeurs ─────────────────────────────────────────────────────────

    SparseTensor() = default;

    // Construction directe depuis coords + features
    SparseTensor(const Eigen::MatrixXi& coords_,
                 const Eigen::MatrixXf& features_,
                 int B, int C, int D, int H, int W)
        : coords(coords_), features(features_),
          batch_size(B), num_channels(C),
          spatial_d(D), spatial_h(H), spatial_w(W)
    {
        validate("SparseTensor::constructor");
    }

    // ── Nombre de voxels actifs ───────────────────────────────────────────────

    int nnz() const { return static_cast<int>(features.rows()); }

    bool empty() const { return nnz() == 0; }

    // ── Encodage / décodage de la clé de hash ─────────────────────────────────

    // Encode (b, d, h, w) en clé uint64 unique dans les dimensions courantes
    uint64_t encode(int b, int d, int h, int w) const {
        return static_cast<uint64_t>(b) * (spatial_d * spatial_h * spatial_w)
             + static_cast<uint64_t>(d) * (spatial_h * spatial_w)
             + static_cast<uint64_t>(h) *  spatial_w
             + static_cast<uint64_t>(w);
    }

    // Décode une clé en (b, d, h, w) — utilisé pour le debug
    void decode(uint64_t key, int& b, int& d, int& h, int& w) const {
        const int dhw = spatial_d * spatial_h * spatial_w;
        const int hw  = spatial_h * spatial_w;
        b = static_cast<int>(key / dhw);
        key %= dhw;
        d = static_cast<int>(key / hw);
        key %= hw;
        h = static_cast<int>(key / spatial_w);
        w = static_cast<int>(key % spatial_w);
    }

    // ── Table de lookup ───────────────────────────────────────────────────────

    // Construit (ou reconstruit) la table coord→index
    void buildLookup() const {
        lookup.clear();
        lookup.reserve(nnz() * 2);
        for (int i = 0; i < nnz(); ++i) {
            const uint64_t key = encode(coords(i,0), coords(i,1),
                                        coords(i,2), coords(i,3));
            lookup[key] = i;
        }
        lookup_valid = true;
    }

    // Invalide la table (à appeler après toute modification de coords)
    void invalidateLookup() { lookup_valid = false; lookup.clear(); }

    // Recherche : retourne l'indice de (b,d,h,w) ou -1 si absent
    int find(int b, int d, int h, int w) const {
        if (!lookup_valid) buildLookup();
        const uint64_t key = encode(b, d, h, w);
        auto it = lookup.find(key);
        return (it != lookup.end()) ? it->second : -1;
    }

    // Vérifie si (b,d,h,w) est un voxel actif
    bool isActive(int b, int d, int h, int w) const {
        return find(b, d, h, w) >= 0;
    }

    // ── Conversion dense → sparse ─────────────────────────────────────────────
    //
    // Crée un SparseTensor depuis un Tensor dense (B, C, D, H, W).
    // Un voxel est actif si max(|x|) sur ses canaux > threshold.
    // La condition porte sur la norme L∞ par canal afin de préserver les
    // voxels actifs même si certains canaux sont négatifs post-ReLU.
    //
    // threshold = 0.0f : conserve tout sauf les voxels strictement nuls
    // threshold > 0.0f : applique un seuil d'activation (recommandé : 1e-4)
    static SparseTensor from_dense(const Tensor& dense, float threshold = 0.0f) {
        if (dense.ndim() != 5)
            throw std::runtime_error("[SparseTensor::from_dense] Attend un Tensor 5D (B,C,D,H,W)");

        const int B = dense.dim(0);
        const int C = dense.dim(1);
        const int D = dense.dim(2);
        const int H = dense.dim(3);
        const int W = dense.dim(4);

        // Passe 1 : collecte des coordonnées actives
        std::vector<std::array<int,4>> active_coords;
        active_coords.reserve(B * D * H * W / 4); // estimation heuristique

        for (int b = 0; b < B; ++b)
        for (int d = 0; d < D; ++d)
        for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            // Norme L∞ sur les canaux
            float max_abs = 0.0f;
            for (int c = 0; c < C; ++c)
                max_abs = std::max(max_abs, std::abs(dense(b, c, d, h, w)));
            if (max_abs > threshold)
                active_coords.push_back({b, d, h, w});
        }

        const int N = static_cast<int>(active_coords.size());

        SparseTensor sp;
        sp.batch_size   = B;
        sp.num_channels = C;
        sp.spatial_d    = D;
        sp.spatial_h    = H;
        sp.spatial_w    = W;
        sp.coords.resize(N, 4);
        sp.features.resize(N, C);

        // Passe 2 : remplissage
        for (int i = 0; i < N; ++i) {
            const auto& ac = active_coords[i];
            sp.coords(i, 0) = ac[0]; // b
            sp.coords(i, 1) = ac[1]; // d
            sp.coords(i, 2) = ac[2]; // h
            sp.coords(i, 3) = ac[3]; // w
            for (int c = 0; c < C; ++c)
                sp.features(i, c) = dense(ac[0], c, ac[1], ac[2], ac[3]);
        }

        sp.lookup_valid = false;
        return sp;
    }

    // ── Conversion sparse → dense ─────────────────────────────────────────────
    //
    // Reconstruit un Tensor dense (B, C, D, H, W) à partir du SparseTensor.
    // Les voxels inactifs sont mis à 0.
    Tensor to_dense() const {
        Tensor dense(batch_size, num_channels, spatial_d, spatial_h, spatial_w);
        dense.setZero();

        for (int i = 0; i < nnz(); ++i) {
            const int b = coords(i, 0);
            const int d = coords(i, 1);
            const int h = coords(i, 2);
            const int w = coords(i, 3);
            for (int c = 0; c < num_channels; ++c)
                dense(b, c, d, h, w) = features(i, c);
        }

        return dense;
    }

    // ── Opérations sur les features ───────────────────────────────────────────

    // Applique ReLU in-place sur les features (max(0, x))
    // Ne modifie pas les coordonnées — les voxels à 0 restent dans la liste
    // (comportement SubManifold : la sparsité est préservée, pas amplifiée)
    void applyReLU() {
        features = features.cwiseMax(0.0f);
        lookup_valid = false; // les features ont changé mais pas les coords
    }

    // Applique Leaky ReLU in-place
    void applyLeakyReLU(float alpha = 0.01f) {
        for (int i = 0; i < features.rows(); ++i)
        for (int j = 0; j < features.cols(); ++j)
            features(i, j) = (features(i, j) > 0.f)
                             ? features(i, j)
                             : alpha * features(i, j);
        lookup_valid = false;
    }

    // Global Average Pooling 3D → Tensor dense (B, C, 1, 1, 1)
    // Utilisé comme transition sparse → dense avant DenseLayer
    Tensor globalAvgPool() const {
        Tensor out(batch_size, num_channels, 1, 1, 1);
        out.setZero();

        // Compteur de voxels actifs par batch
        Eigen::VectorXi counts = Eigen::VectorXi::Zero(batch_size);

        for (int i = 0; i < nnz(); ++i) {
            const int b = coords(i, 0);
            ++counts[b];
            for (int c = 0; c < num_channels; ++c)
                out(b, c, 0, 0, 0) += features(i, c);
        }

        // Normalisation par le nombre de voxels actifs (ou total si 0)
        for (int b = 0; b < batch_size; ++b) {
            const float denom = (counts[b] > 0)
                               ? static_cast<float>(counts[b])
                               : static_cast<float>(spatial_d * spatial_h * spatial_w);
            for (int c = 0; c < num_channels; ++c)
                out(b, c, 0, 0, 0) /= denom;
        }

        return out;
    }

    // ── Densité ───────────────────────────────────────────────────────────────

    float density() const {
        const int total = batch_size * spatial_d * spatial_h * spatial_w;
        return (total > 0) ? static_cast<float>(nnz()) / total : 0.f;
    }

    // ── Affichage / debug ─────────────────────────────────────────────────────

    void printShape() const {
        std::cout << "SparseTensor ("
                  << batch_size << ", " << num_channels << ", "
                  << spatial_d  << ", " << spatial_h    << ", " << spatial_w << ")"
                  << "  nnz=" << nnz()
                  << "  density=" << std::fixed << std::setprecision(2)
                  << density() * 100.f << "%" << std::endl;
    }

    void printStats() const {
        printShape();
        if (nnz() == 0) { std::cout << "  [vide]" << std::endl; return; }

        float fmin =  std::numeric_limits<float>::max();
        float fmax = -std::numeric_limits<float>::max();
        float fsum = 0.f;
        for (int i = 0; i < features.rows(); ++i)
        for (int j = 0; j < features.cols(); ++j) {
            fmin = std::min(fmin, features(i,j));
            fmax = std::max(fmax, features(i,j));
            fsum += features(i,j);
        }
        const float fmean = fsum / (features.rows() * features.cols());
        std::cout << "  Features — min: " << fmin
                  << "  max: " << fmax
                  << "  mean: " << fmean << std::endl;

        // Distribution par batch
        Eigen::VectorXi cnt = Eigen::VectorXi::Zero(batch_size);
        for (int i = 0; i < nnz(); ++i) ++cnt[coords(i,0)];
        std::cout << "  Voxels actifs par batch: [";
        for (int b = 0; b < batch_size; ++b) {
            std::cout << cnt[b];
            if (b + 1 < batch_size) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Affiche les N premiers voxels actifs
    void printSample(int n = 5) const {
        std::cout << "SparseTensor sample (" << std::min(n, nnz()) << "/" << nnz() << " voxels):" << std::endl;
        for (int i = 0; i < std::min(n, nnz()); ++i) {
            std::cout << "  [" << coords(i,0) << "," << coords(i,1)
                      << "," << coords(i,2) << "," << coords(i,3) << "]"
                      << " features=[ ";
            for (int c = 0; c < std::min(4, num_channels); ++c)
                std::cout << std::setprecision(4) << features(i,c) << " ";
            if (num_channels > 4) std::cout << "...";
            std::cout << "]" << std::endl;
        }
    }

private:

    // ── Validation interne ────────────────────────────────────────────────────

    void validate(const std::string& caller) const {
        if (coords.rows() != features.rows())
            throw std::runtime_error("[" + caller + "] coords.rows() != features.rows()");
        if (coords.cols() != 4)
            throw std::runtime_error("[" + caller + "] coords doit avoir 4 colonnes (b,d,h,w)");
        if (features.cols() != num_channels)
            throw std::runtime_error("[" + caller + "] features.cols() != num_channels");
    }
};
