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

    // --- Interface publique ---

    void setTargets(const Tensor& target) {
        targets_cache = target;
    }

    float getCurrentLoss() {
        assertForwardCalled("getCurrentLoss");
        return computeLoss(predictions_cache, targets_cache);
    }

    // --- Surcharges Layer ---

    Tensor forward(const Tensor& input) override {
        predictions_cache = input;
        forward_called = true;
        return input;
    }

    Tensor backward(const Tensor& grad_output) override {
        assertForwardCalled("backward");
        Tensor grad = computeGradient(predictions_cache, targets_cache);
        forward_called = false;
        return grad;
    }

    std::string getName() const override = 0;

protected:
    // --- Interface à implémenter par les sous-classes ---

    virtual float computeLoss(const Tensor& pred, const Tensor& target) = 0;
    virtual Tensor computeGradient(const Tensor& pred, const Tensor& target) = 0;

    void assertForwardCalled(const std::string& caller) const {
        if (!forward_called) {
            throw std::runtime_error(getName() + ": forward() doit être appelé avant " + caller + "()");
        }
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
        const int batch_size  = pred.dim(0);
        const int num_classes = pred.dim(1);
        const float epsilon   = 1e-12f;
        float total_loss      = 0.0f;

        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < num_classes; ++c) {
                if (target(b, c, 0, 0) > 0.5f) { // encodage one-hot
                    float prob = std::max(pred(b, c, 0, 0), epsilon);
                    total_loss += -std::log(prob);
                }
            }
        }

        return total_loss / batch_size;
    }

    // Gradient combiné softmax + cross-entropy : (p - y) / batch_size
    Tensor computeGradient(const Tensor& pred, const Tensor& target) override {
        const int batch_size  = pred.dim(0);
        const int num_classes = pred.dim(1);
        Tensor grad(batch_size, num_classes, 1, 1);

#pragma omp parallel for collapse(2)
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < num_classes; ++c) {
                grad(b, c, 0, 0) = (pred(b, c, 0, 0) - target(b, c, 0, 0)) / batch_size;
            }
        }

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
        const int total      = pred.size();
        float     total_loss = 0.0f;

        for (int i = 0; i < total; ++i) {
            float diff = pred[i] - target[i];
            total_loss += diff * diff;
        }

        return total_loss / (2.0f * pred.dim(0));
    }

    // Gradient : (pred - target) / batch_size
    Tensor computeGradient(const Tensor& pred, const Tensor& target) override {
        const int batch_size = pred.dim(0);
        Tensor    grad(pred.shape());

        for (int i = 0; i < grad.size(); ++i) {
            grad[i] = (pred[i] - target[i]) / batch_size;
        }

        return grad;
    }
};


// =============================================================================
// Softmax + Cross-Entropy fusionnés (évite les instabilités numériques)
// Intègre le calcul softmax dans la passe forward pour un gradient plus stable
// =============================================================================
class SoftmaxCrossEntropyLayer : public LossLayer {
private:
    Tensor softmax_probs; // probabilités softmax, nécessaires au backward

public:
    SoftmaxCrossEntropyLayer() = default;
    ~SoftmaxCrossEntropyLayer() override = default;

    std::string getName() const override { return "SoftmaxCrossEntropyLayer"; }

    // Forward : applique softmax et met en cache les probabilités
    Tensor forward(const Tensor& input) override {
        if (input.dim(2) != 1 || input.dim(3) != 1) {
            throw std::runtime_error(getName() + ": forme attendue [batch, classes, 1, 1]");
        }

        const int batch_size  = input.dim(0);
        const int num_classes = input.dim(1);
        softmax_probs = Tensor(batch_size, num_classes, 1, 1);

#pragma omp parallel for
        for (int b = 0; b < batch_size; ++b) {
            // Stabilité numérique : soustraction du maximum
            float max_val = -std::numeric_limits<float>::infinity();
            for (int c = 0; c < num_classes; ++c) {
                max_val = std::max(max_val, input(b, c, 0, 0));
            }

            float sum_exp = 0.0f;
            for (int c = 0; c < num_classes; ++c) {
                float exp_val = std::exp(input(b, c, 0, 0) - max_val);
                softmax_probs(b, c, 0, 0) = exp_val;
                sum_exp += exp_val;
            }

            for (int c = 0; c < num_classes; ++c) {
                softmax_probs(b, c, 0, 0) /= sum_exp;
            }
        }

        // Synchronise predictions_cache avec LossLayer::getCurrentLoss
        predictions_cache = softmax_probs;
        forward_called    = true;

        return input; // retourne les logits pour la chaîne
    }

    // --- Utilitaires ---

    // Indice de la classe prédite pour chaque élément du batch
    Tensor getPredictions() const {
        assertForwardCalled("getPredictions");
        const int batch_size  = softmax_probs.dim(0);
        const int num_classes = softmax_probs.dim(1);
        Tensor    predictions(batch_size, 1, 1, 1);

        for (int b = 0; b < batch_size; ++b) {
            int   max_class = 0;
            float max_prob  = -1.0f;

            for (int c = 0; c < num_classes; ++c) {
                if (softmax_probs(b, c, 0, 0) > max_prob) {
                    max_prob  = softmax_probs(b, c, 0, 0);
                    max_class = c;
                }
            }
            predictions(b, 0, 0, 0) = static_cast<float>(max_class);
        }

        return predictions;
    }

    // Précision sur un batch (targets one-hot dans targets_cache)
    float computeAccuracy() const {
        assertForwardCalled("computeAccuracy");
        Tensor    preds       = getPredictions();
        const int batch_size  = preds.dim(0);
        const int num_classes = targets_cache.dim(1);
        int       correct     = 0;

        for (int b = 0; b < batch_size; ++b) {
            int true_class = -1;
            for (int c = 0; c < num_classes; ++c) {
                if (targets_cache(b, c, 0, 0) > 0.5f) {
                    true_class = c;
                    break;
                }
            }
            if (static_cast<int>(preds(b, 0, 0, 0)) == true_class) {
                ++correct;
            }
        }

        return static_cast<float>(correct) / batch_size;
    }

protected:
    float computeLoss(const Tensor& pred, const Tensor& target) override {
        // pred == softmax_probs (via predictions_cache)
        const int   batch_size  = pred.dim(0);
        const int   num_classes = pred.dim(1);
        const float epsilon     = 1e-12f;
        float       total_loss  = 0.0f;

        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < num_classes; ++c) {
                if (target(b, c, 0, 0) > 0.5f) {
                    total_loss += -std::log(std::max(pred(b, c, 0, 0), epsilon));
                }
            }
        }

        return total_loss / batch_size;
    }

    // Gradient combiné softmax + cross-entropy : (softmax(x) - y) / batch_size
    Tensor computeGradient(const Tensor& pred, const Tensor& target) override {
        const int batch_size  = softmax_probs.dim(0);
        const int num_classes = softmax_probs.dim(1);
        Tensor    grad(batch_size, num_classes, 1, 1);

#pragma omp parallel for collapse(2)
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < num_classes; ++c) {
                grad(b, c, 0, 0) = (softmax_probs(b, c, 0, 0) - target(b, c, 0, 0)) / batch_size;
            }
        }

        return grad;
    }
};