# pragma once
# include "Layer.hpp"

class LossLayer : public Layer {

protected:
    Tensor predictions;
    Tensor targets;

public:
    LossLayer() = default;
    virtual ~LossLayer() = default;

    virtual float computeLoss(const Tensor& pred, const Tensor& target) = 0;
    virtual Tensor computeGradient(const Tensor& pred, const Tensor& target) = 0;

    // Forward pass: calcule et retourne la perte
    Tensor forward(const Tensor& input) override {
        predictions = input; // Les predictions viennent de la derniere couche
        return input; // Pour la chaine forward, on retourne juste l'input
    }

    // Backward pass: clacule le gradient de la perte
    Tensor backward(const Tensor& gradOutput) override {
        return computeGradient(predictions, targets);
    }

    virtual void setTargets(const Tensor& target) {
        targets = target;
    }

    virtual float getCurrentLoss() {
        return computeLoss(predictions, targets);
    }

    std::string getName() const override = 0;

};

class CrossEntropyLoss : public LossLayer {
private:
    Tensor predictions_cache;
    Tensor targets_cache;
    bool forward_called = false;


public:
    CrossEntropyLoss() = default;

    std::string getName() const override { return "CrossEntropy"; }

    Tensor forward(const Tensor& input) override {

        predictions_cache = input;

        forward_called = true;

        return input;
    }

    Tensor backward(const Tensor& grad_output) override {
        if (!forward_called) {
            throw std::runtime_error("Forward must be called before backward");
        }

        int batch_size = predictions_cache.dim(0);
        int num_classes = predictions_cache.dim(1);
        // std::cout << "crossentropy : " << batch_size << " x " << num_classes << std::endl;
        // predictions_cache.printShape();
        Tensor grad_input(batch_size, num_classes, 1, 1);

        // grad_input.printShape();

        // Gradient pour softmax + cross-entropy: p - y
# pragma omp parallel for collapse(2)
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < num_classes; c++) {
                grad_input(b, c, 0, 0) = (predictions_cache(b, c, 0, 0) - targets_cache(b, c, 0, 0)) / batch_size;
            }
        }

        forward_called = false;

        // grad_input.printShape();
        return grad_input;
    }

    float computeLoss(const Tensor& pred, const Tensor& target) override {

        predictions_cache = pred;
        targets_cache = target;

        int batch_size = pred.dim(0);
        int num_classes = pred.dim(1);
        float total_loss = 0.0f;
        const float epsilon = 1e-12f;

        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < num_classes; c++) {
                if (target(b, c, 0, 0) > 0.5f) { // One-hot
                    // std::cout << "target : " << target(b, c, 0, 0) ;
                    // std::cout << " | prediction : " << pred(b, c, 0, 0) << std::endl;
                    float prob = std::max(pred(b, c, 0, 0), epsilon);
                    total_loss += -std::log(prob);
                    // std::cout << "prob : " << prob << " |  total_loss : " << total_loss << std::endl;
                }
            }
        }


        return total_loss / batch_size;
    }

    Tensor computeGradient(const Tensor& pred, const Tensor& target) override {
        predictions_cache = pred;
        targets_cache = target;
        return backward(Tensor()); // Appelle bacward interne
    }

    void setTargets(const Tensor& target) override {
        targets_cache = target;
    }

    float getCurrentLoss() override {
        // std::cout << "Getting current loss" << std::endl;
        // predictions_cache.printShape();
        return computeLoss(predictions_cache, targets_cache);
    }

};


class MSELoss : public LossLayer {
public:
    MSELoss() = default;

    std::string getName() const override { return "MSELoss"; }

    float computeLoss(const Tensor& pred, const Tensor& target) {
        int total_elements = pred.size();
        float total_loss = 0.0f;

        for (int i = 0; i < total_elements; i++) {
            float diff = pred[i] - target[i];
            total_loss += diff * diff;
        }

        return total_loss / (2.0f * pred.dim(0)); // Moyenne sur le batch
    }

    Tensor computeGradient(const Tensor& pred, const Tensor& target) override {
        Tensor grad(pred.shape());

        // Gradient: pred - target
        for (int i = 0; i < grad.size(); i++) {
            grad[i] = (pred[i] - target[i]) / pred.dim(0); //Normalise par batch size
        }

        return grad;
    }
};


class SoftmaxCrossEntropyLayer : public Layer {
private:
    Tensor probabilities;  // Probabilités softmax
    Tensor targets;        // Cibles (one-hot encoding)
    bool has_targets = false;

public:
    SoftmaxCrossEntropyLayer() = default;
    ~SoftmaxCrossEntropyLayer() override = default;

    // Forward pass: retourne les logits (non modifiés) pour la chaîne
    // La perte est calculée séparément
    Tensor forward(const Tensor& input) override {
        int batch_size = input.dim(0);
        int num_classes = input.dim(1);

        if (input.dim(2) != 1 || input.dim(3) != 1) {
            throw std::runtime_error("SoftmaxCrossEntropyLayer: Shape attendue [batch, classes, 1, 1]");
        }

        probabilities = Tensor(batch_size, num_classes, 1, 1);

#pragma omp parallel for
        for (int b = 0; b < batch_size; ++b) {
            // Stabilité numérique: soustraire le maximum
            float max_val = -std::numeric_limits<float>::infinity();
            for (int c = 0; c < num_classes; ++c) {
                max_val = std::max(max_val, input(b, c, 0, 0));
            }

            // Calculer softmax
            float sum_exp = 0.0f;
            for (int c = 0; c < num_classes; ++c) {
                float exp_val = std::exp(input(b, c, 0, 0) - max_val);
                sum_exp += exp_val;
                probabilities(b, c, 0, 0) = exp_val;
            }

            // Normalisation
            for (int c = 0; c < num_classes; ++c) {
                probabilities(b, c, 0, 0) /= sum_exp;
            }
        }

        return input; // Retourne les logits pour la chaîne
    }

    // Backward pass: calcule directement le gradient softmax + cross-entropy
    Tensor backward(const Tensor& grad_output) override {
        if (!has_targets) {
            throw std::runtime_error("SoftmaxCrossEntropyLayer: targets non définis");
        }

        int batch_size = probabilities.dim(0);
        int num_classes = probabilities.dim(1);
        Tensor grad_input(batch_size, num_classes, 1, 1);

        // Gradient combiné: softmax - target
#pragma omp parallel for collapse(2)
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < num_classes; ++c) {
                grad_input(b, c, 0, 0) = (probabilities(b, c, 0, 0) - targets(b, c, 0, 0)) / batch_size;
            }
        }

        return grad_input;
    }

    // Définir les cibles pour le calcul de la perte
    void setTargets(const Tensor& target_tensor)  {
        if (target_tensor.shape() != probabilities.shape()) {
            throw std::runtime_error("SoftmaxCrossEntropyLayer: Shape des cibles incompatible");
        }
        targets = target_tensor;
        has_targets = true;
    }

    // Calculer la perte cross-entropy
    float computeLoss() const {

        if (!has_targets) {
            throw std::runtime_error("SoftmaxCrossEntropyLayer: targets non définis");
        }

        int batch_size = probabilities.dim(0);
        int num_classes = probabilities.dim(1);
        float total_loss = 0.0f;
        const float epsilon = 1e-12f; // Pour éviter log(0)

        for (int b = 0; b < batch_size; ++b) {
            float sample_loss = 0.0f;
            for (int c = 0; c < num_classes; ++c) {
                if (targets(b, c, 0, 0) > 0.5f) { // One-hot encoding
                    float prob = std::max(probabilities(b, c, 0, 0), epsilon);
                    sample_loss += -std::log(prob);
                }
            }
            total_loss += sample_loss;
        }
        return total_loss / batch_size;
    }

    // Obtenir les prédictions (indices des classes)
    Tensor getPredictions() const {
        int batch_size = probabilities.dim(0);
        int num_classes = probabilities.dim(1);
        Tensor predictions(batch_size, 1, 1, 1);

        for (int b = 0; b < batch_size; ++b) {
            int max_class = 0;
            float max_prob = 0.0f;

            for (int c = 0; c < num_classes; ++c) {
                if (probabilities(b, c, 0, 0) > max_prob) {
                    max_prob = probabilities(b, c, 0, 0);
                    max_class = c;
                }
            }

            predictions(b, 0, 0, 0) = static_cast<float>(max_class);
        }

        return predictions;
    }

    // Calculer la précision
    float computeAccuracy(const Tensor& true_labels) const {
        Tensor predictions = getPredictions();
        int batch_size = predictions.dim(0);
        int correct = 0;

        for (int b = 0; b < batch_size; ++b) {
            // Vérifier quelle classe est active dans les cibles one-hot
            int true_class = -1;
            for (int c = 0; c < targets.dim(1); ++c) {
                if (targets(b, c, 0, 0) > 0.5f) {
                    true_class = c;
                    break;
                }
            }

            if (static_cast<int>(predictions(b, 0, 0, 0)) == true_class) {
                correct++;
            }
        }

        return static_cast<float>(correct) / batch_size;
    }

    std::string getName() const override {
        return "SoftmaxCrossEntropy";
    }
};