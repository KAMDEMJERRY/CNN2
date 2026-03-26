// Augmentation3D.hpp
// =============================================================================
// Module d'augmentation de données pour volumes 3D (C, D, H, W).
//
// Transformations disponibles :
//   1. Flip  — axe D, H ou W, indépendamment et aléatoirement
//   2. Rotation 90° — dans le plan axial (autour de l'axe D) : n × 90°
//   3. Bruit gaussien — ajout de N(0, σ²) indépendant par voxel
//   4. Scaling d'intensité — multiplication globale par un facteur ∈ [a, b]
//
// Usage minimal :
//   AugmentConfig cfg;                 // valeurs par défaut raisonnables
//   Augmentor3D   aug(cfg);
//   Tensor out = aug.apply(vol, rng);  // vol : Tensor(1, C, D, H, W)
// =============================================================================

#pragma once
#include "Tensor.hpp"
#include <random>
#include <cmath>
#include <cstring>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

struct AugmentConfig {
    // --- Flip ---
    float flip_prob_d    = 0.5f;  // probabilité de flip selon l'axe profondeur
    float flip_prob_h    = 0.5f;  // probabilité de flip selon l'axe hauteur
    float flip_prob_w    = 0.5f;  // probabilité de flip selon l'axe largeur

    // --- Rotation axiale (autour de l'axe D) ---
    float rotate_prob    = 0.5f;  // probabilité d'appliquer une rotation
    // La rotation est choisie parmi {90°, 180°, 270°} si activée

    // --- Bruit gaussien ---
    float noise_prob     = 0.5f;  // probabilité d'ajouter du bruit
    float noise_std_min  = 0.01f; // borne inf de σ (tirée uniformément)
    float noise_std_max  = 0.08f; // borne sup de σ

    // --- Scaling d'intensité ---
    float scale_prob     = 0.5f;  // probabilité de scaler les intensités
    float scale_min      = 0.90f; // facteur min
    float scale_max      = 1.10f; // facteur max
};

// ─────────────────────────────────────────────────────────────────────────────
// Augmentor3D
// ─────────────────────────────────────────────────────────────────────────────

class Augmentor3D {
public:

    explicit Augmentor3D(const AugmentConfig& cfg = AugmentConfig{})
        : cfg_(cfg) {}

    // ── Point d'entrée principal ──────────────────────────────────────────────
    //
    // vol : Tensor(1, C, D, H, W) — volume individuel issu du dataset
    // rng : générateur fourni par le DataLoader (thread-safe si externe)
    //
    // Retourne un Tensor de même shape et rang logique.

    Tensor apply(const Tensor& vol, std::mt19937& rng) const {
        if (vol.ndim() != 5)
            throw std::runtime_error("[Augmentor3D] apply() requiert un Tensor 5D (1,C,D,H,W)");
        if (vol.dim(0) != 1)
            throw std::runtime_error("[Augmentor3D] apply() requiert batch=1 par appel");

        Tensor out = vol;  // copie de travail

        // 1. Flips
        if (bernoulli(rng, cfg_.flip_prob_d)) out = flipAxis(out, /*axis=*/2);
        if (bernoulli(rng, cfg_.flip_prob_h)) out = flipAxis(out, /*axis=*/3);
        if (bernoulli(rng, cfg_.flip_prob_w)) out = flipAxis(out, /*axis=*/4);

        // 2. Rotation 90° axiale (plan H×W, D fixé)
        if (bernoulli(rng, cfg_.rotate_prob)) {
            int k = std::uniform_int_distribution<int>(1, 3)(rng); // 1, 2 ou 3 fois
            for (int i = 0; i < k; ++i)
                out = rotate90HW(out);
        }

        // 3. Bruit gaussien
        if (bernoulli(rng, cfg_.noise_prob)) {
            float sigma = std::uniform_real_distribution<float>(
                cfg_.noise_std_min, cfg_.noise_std_max)(rng);
            out = addGaussianNoise(out, sigma, rng);
        }

        // 4. Scaling d'intensité
        if (bernoulli(rng, cfg_.scale_prob)) {
            float scale = std::uniform_real_distribution<float>(
                cfg_.scale_min, cfg_.scale_max)(rng);
            out = scaleIntensity(out, scale);
        }

        return out;
    }

    const AugmentConfig& config() const { return cfg_; }

private:

    AugmentConfig cfg_;

    // ── Helpers stochastiques ─────────────────────────────────────────────────

    static bool bernoulli(std::mt19937& rng, float prob) {
        return std::bernoulli_distribution(static_cast<double>(prob))(rng);
    }

    // ── 1. Flip selon un axe interne (0=B, 1=C, 2=D, 3=H, 4=W) ─────────────

    static Tensor flipAxis(const Tensor& t, int axis) {
        const int B = t.dim(0);
        const int C = t.dim(1);
        const int D = t.dim(2);
        const int H = t.dim(3);
        const int W = t.dim(4);

        Tensor out(B, C, D, H, W);

        for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
        for (int d = 0; d < D; ++d)
        for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            int fb = (axis == 0) ? (B - 1 - b) : b;
            int fc = (axis == 1) ? (C - 1 - c) : c;
            int fd = (axis == 2) ? (D - 1 - d) : d;
            int fh = (axis == 3) ? (H - 1 - h) : h;
            int fw = (axis == 4) ? (W - 1 - w) : w;
            out(b, c, d, h, w) = t(fb, fc, fd, fh, fw);
        }
        return out;
    }

    // ── 2. Rotation 90° dans le plan H×W (D inchangé) ────────────────────────
    //
    // Convention : rotation dans le sens trigonométrique.
    // (h, w) → (W-1-w, h)   après transposition + flip W
    //
    // Pour un volume (B, C, D, H, W) :
    //   out(b,c,d, w, H-1-h) = in(b,c,d,h,w)
    //   → shape de sortie (B, C, D, W, H)  si H≠W
    //   → shape identique si H == W        (cas FractureMNIST3D : 28==28)

    static Tensor rotate90HW(const Tensor& t) {
        const int B = t.dim(0);
        const int C = t.dim(1);
        const int D = t.dim(2);
        const int H = t.dim(3);
        const int W = t.dim(4);

        // sortie : (B, C, D, W, H)
        Tensor out(B, C, D, W, H);

        for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
        for (int d = 0; d < D; ++d)
        for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
            out(b, c, d, w, H - 1 - h) = t(b, c, d, h, w);

        return out;
    }

    // ── 3. Bruit gaussien ─────────────────────────────────────────────────────

    static Tensor addGaussianNoise(const Tensor& t, float sigma, std::mt19937& rng) {
        Tensor out = t;
        std::normal_distribution<float> dist(0.0f, sigma);
        float* data = out.getData();
        int    n    = out.size();
        for (int i = 0; i < n; ++i)
            data[i] += dist(rng);
        return out;
    }

    // ── 4. Scaling d'intensité ────────────────────────────────────────────────

    static Tensor scaleIntensity(const Tensor& t, float scale) {
        Tensor out = t;
        float* data = out.getData();
        int    n    = out.size();
        for (int i = 0; i < n; ++i)
            data[i] *= scale;
        return out;
    }
};
