#pragma once
#include "Layer.hpp"

// =============================================================================
// Classe de base abstraite pour toutes les couches de perte
// =============================================================================
class LossLayer : public Layer {
protected:
    Tensor predictions_cache;
    Tensor targets_cache;
    bool forward_called = false;

public:
    LossLayer() = default;
    virtual ~LossLayer() = default;

    void setTargets(const Tensor& target) { targets_cache = target; }

    float getCurrentLoss() {
        assertForwardCalled("getCurrentLoss");
        return computeLoss(predictions_cache, targets_cache);
    }

    Tensor forward(const Tensor& input) override {
        predictions_cache = input;
        forward_called    = true;
        return input;
    }

    Tensor backward(const Tensor& grad_output) override {
        assertForwardCalled("backward");
        Tensor grad   = computeGradient(predictions_cache, targets_cache);
        forward_called = false;
        return grad;
    }

    std::string getName() const override = 0;

protected:
    virtual float  computeLoss    (const Tensor& pred, const Tensor& target) = 0;
    virtual Tensor computeGradient(const Tensor& pred, const Tensor& target) = 0;

    void assertForwardCalled(const std::string& caller) const {
        if (!forward_called)
            throw std::runtime_error(
                getName() + ": forward() doit être appelé avant " + caller + "()");
    }
};


// =============================================================================
// Cross-Entropy Loss (attend des probabilités softmax en entrée)
// =============================================================================
class CrossEntropyLoss : public LossLayer {
public:
    CrossEntropyLoss() = default;
    std::string getName() const override { return "CrossEntropyLoss"; }

protected:
    float computeLoss(const Tensor& pred, const Tensor& target) override {
        const int   B       = pred.dim(0);
        const int   C       = pred.dim(1);
        const float epsilon = 1e-12f;
        float loss = 0.f;
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C; ++c)
                if (target(b, c, 0, 0) > 0.5f)
                    loss += -std::log(std::max(pred(b, c, 0, 0), epsilon));
        return loss / B;
    }

    Tensor computeGradient(const Tensor& pred, const Tensor& target) override {
        const int B = pred.dim(0);
        const int C = pred.dim(1);
        Tensor grad(B, C, 1, 1);
#pragma omp parallel for collapse(2)
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C; ++c)
                grad(b, c, 0, 0) = (pred(b, c, 0, 0) - target(b, c, 0, 0)) / B;
        return grad;
    }
};


// =============================================================================
// MSE Loss
// =============================================================================
class MSELoss : public LossLayer {
public:
    MSELoss() = default;
    std::string getName() const override { return "MSELoss"; }

protected:
    float computeLoss(const Tensor& pred, const Tensor& target) override {
        float loss = 0.f;
        for (int i = 0; i < pred.size(); ++i) {
            float d = pred[i] - target[i];
            loss += d * d;
        }
        return loss / (2.f * pred.dim(0));
    }

    Tensor computeGradient(const Tensor& pred, const Tensor& target) override {
        const int B = pred.dim(0);
        Tensor grad(pred.shape());
        for (int i = 0; i < grad.size(); ++i)
            grad[i] = (pred[i] - target[i]) / B;
        return grad;
    }
};


// =============================================================================
// SoftmaxCrossEntropyLayer — Softmax + Weighted Cross-Entropy + Focal Loss
//
// Loss :
//   WCE  (gamma=0) : L = -w_t * log(p_t)
//   FL   (gamma>0) : L = -w_t * (1-p_t)^gamma * log(p_t)
//
// Gradient par rapport aux logits z_k (avant softmax) — dérivation exacte :
//
//   Notons : p_t = softmax(z)_t,  w = class_weight[t]
//
//   WCE (gamma=0) :
//     dL/dz_k = w * (p_k - y_k) / B        [forme compacte standard]
//
//   Focal Loss (gamma>0) :
//     k == t : w * [gamma*p_t*(1-p_t)^gamma*log(p_t) - (1-p_t)^(gamma+1)] / B
//     k != t : w * p_k * (1-p_t)^(gamma-1) * [(1-p_t) - gamma*p_t*log(p_t)] / B
//
//   Ces formules se réduisent bien à w*(p_k - y_k)/B quand gamma→0 :
//     k == t : (1-p_t)^1 * (... → 0) - (1-p_t)^1 = -(1-p_t) = p_t - 1 = p_t - y_t  ✓
//     k != t : p_k*(1-p_t)^0*[(1-p_t) - 0] = p_k*(1-p_t) ≈ p_k - 0 = p_k - y_k    ✓
//
// Instabilité gamma<1 : (1-p_t)^(gamma-1) → ∞ quand p_t→1, clampé à 1e-6.
// =============================================================================
class SoftmaxCrossEntropyLayer : public LossLayer {
private:
    Tensor             softmax_probs_;
    std::vector<float> class_weights_;
    float              gamma_;

public:
    // ── Constructeurs ─────────────────────────────────────────────────────────
    SoftmaxCrossEntropyLayer()
        : gamma_(0.f) {}

    explicit SoftmaxCrossEntropyLayer(const std::vector<float>& weights,
                                       float gamma = 0.f)
        : class_weights_(weights), gamma_(gamma) {}

    ~SoftmaxCrossEntropyLayer() override = default;

    // ── Setters ───────────────────────────────────────────────────────────────
    void setClassWeights(const std::vector<float>& w) { class_weights_ = w; }
    void setGamma(float g)                             { gamma_ = g;        }

    std::string getName() const override { return "SoftmaxCrossEntropyLayer"; }

    // ── Forward : softmax + cache ─────────────────────────────────────────────
    Tensor forward(const Tensor& input) override {
        if (input.dim(2) != 1 || input.dim(3) != 1)
            throw std::runtime_error(
                getName() + ": forme attendue (B, C, 1, 1), reçu dim2="
                + std::to_string(input.dim(2)));

        const int B = input.dim(0);
        const int C = input.dim(1);
        softmax_probs_ = Tensor(B, C, 1, 1);

#pragma omp parallel for
        for (int b = 0; b < B; ++b) {
            // Stabilité : soustraction du max avant exp
            float max_val = -std::numeric_limits<float>::infinity();
            for (int c = 0; c < C; ++c)
                max_val = std::max(max_val, input(b, c, 0, 0));

            float sum_exp = 0.f;
            for (int c = 0; c < C; ++c) {
                float e = std::exp(input(b, c, 0, 0) - max_val);
                softmax_probs_(b, c, 0, 0) = e;
                sum_exp += e;
            }
            for (int c = 0; c < C; ++c)
                softmax_probs_(b, c, 0, 0) /= sum_exp;
        }

        predictions_cache = softmax_probs_;
        forward_called    = true;
        return input;  // retourne les logits — la loss lit predictions_cache
    }

    // ── Utilitaires ───────────────────────────────────────────────────────────

    // Argmax des probabilités softmax → classe prédite par sample
    Tensor getPredictions() const {
        assertForwardCalled("getPredictions");
        const int B = softmax_probs_.dim(0);
        const int C = softmax_probs_.dim(1);
        Tensor out(B, 1, 1, 1);
        for (int b = 0; b < B; ++b) {
            int   best_c = 0;
            float best_p = -1.f;
            for (int c = 0; c < C; ++c)
                if (softmax_probs_(b, c, 0, 0) > best_p) {
                    best_p = softmax_probs_(b, c, 0, 0);
                    best_c = c;
                }
            out(b, 0, 0, 0) = static_cast<float>(best_c);
        }
        return out;
    }

    // Accuracy sur le batch courant (targets one-hot)
    float computeAccuracy() const {
        assertForwardCalled("computeAccuracy");
        const int B = softmax_probs_.dim(0);
        const int C = targets_cache.dim(1);
        Tensor preds = getPredictions();
        int correct  = 0;
        for (int b = 0; b < B; ++b) {
            int true_c = -1;
            for (int c = 0; c < C; ++c)
                if (targets_cache(b, c, 0, 0) > 0.5f) { true_c = c; break; }
            if (static_cast<int>(preds(b, 0, 0, 0)) == true_c) ++correct;
        }
        return static_cast<float>(correct) / B;
    }

protected:
    // ── Loss ──────────────────────────────────────────────────────────────────
    float computeLoss(const Tensor& pred, const Tensor& target) override {
        const int   B       = pred.dim(0);
        const int   C       = pred.dim(1);
        const float epsilon = 1e-12f;
        float loss = 0.f;

        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C; ++c) {
                if (target(b, c, 0, 0) <= 0.5f) continue;

                const float w   = class_weights_.empty() ? 1.f : class_weights_[c];
                const float p_t = std::max(pred(b, c, 0, 0), epsilon);

                if (gamma_ < 1e-6f) {
                    loss += -w * std::log(p_t);
                } else {
                    const float omp = std::max(1.f - p_t, epsilon);
                    loss += -w * std::pow(omp, gamma_) * std::log(p_t);
                }
            }

        return loss / B;
    }

    // ── Gradient ──────────────────────────────────────────────────────────────
    Tensor computeGradient(const Tensor& pred, const Tensor& target) override {
        const int   B       = softmax_probs_.dim(0);
        const int   C       = softmax_probs_.dim(1);
        const float epsilon = 1e-12f;
        Tensor grad(B, C, 1, 1);

#pragma omp parallel for
        for (int b = 0; b < B; ++b) {
            // Trouve la vraie classe et son poids
            int   true_c = -1;
            float w      = 1.f;
            for (int c = 0; c < C; ++c)
                if (target(b, c, 0, 0) > 0.5f) {
                    true_c = c;
                    if (!class_weights_.empty()) w = class_weights_[c];
                    break;
                }

            // Cible introuvable : gradient nul (ne devrait pas arriver)
            if (true_c == -1) {
                for (int c = 0; c < C; ++c) grad(b, c, 0, 0) = 0.f;
                continue;
            }

            const float p_t    = std::max(softmax_probs_(b, true_c, 0, 0), epsilon);
            const float omp    = std::max(1.f - p_t, epsilon);  // (1 - p_t)
            const float log_pt = std::log(p_t);                  // ≤ 0

            if (gamma_ < 1e-6f) {
                // ── WCE standard : w * (p_k - y_k) / B ───────────────────────
                for (int c = 0; c < C; ++c) {
                    const float y_k = target(b, c, 0, 0);
                    grad(b, c, 0, 0) =
                        w * (softmax_probs_(b, c, 0, 0) - y_k) / B;
                }
            } else {
                // ── Focal Loss — gradient exact ───────────────────────────────
                // Clamp pour éviter (1-p_t)^(gamma-1) → ∞ quand p_t → 1
                const float omp_safe = std::max(omp, 1e-6f);

                const float pw_g   = std::pow(omp,      gamma_);        // (1-p_t)^gamma
                const float pw_gm1 = std::pow(omp_safe, gamma_ - 1.f); // (1-p_t)^(gamma-1)
                const float pw_gp1 = pw_g * omp;                        // (1-p_t)^(gamma+1)

                // Facteur commun aux classes k != t
                const float common = pw_gm1 * (omp - gamma_ * p_t * log_pt);

                for (int c = 0; c < C; ++c) {
                    const float p_k = softmax_probs_(b, c, 0, 0);
                    float g;
                    if (c == true_c)
                        // k == t : gamma*p_t*(1-p_t)^gamma*log(p_t) - (1-p_t)^(gamma+1)
                        g = gamma_ * p_t * pw_g * log_pt - pw_gp1;
                    else
                        // k != t : p_k * (1-p_t)^(gamma-1) * [(1-p_t) - gamma*p_t*log(p_t)]
                        g = p_k * common;

                    grad(b, c, 0, 0) = w * g / B;
                }
            }
        }
        return grad;
    }
};