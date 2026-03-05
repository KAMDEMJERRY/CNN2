// Optimizer.hpp
# pragma once
# include "Tensor.hpp"
# include <Eigen/Dense>
# include <memory>

class Optimizer {
protected:
    float learning_rate;
    float clip_norm = 0.0f;  // 0 = pas de clipping


public:
    Optimizer(float lr = 0.01f) : learning_rate(lr) {}
    virtual ~Optimizer() = default;

    // Pour les couches convolutives
    virtual void updateWeights(Tensor& weights, const Tensor& gradients) = 0;
    virtual void updateBias(Eigen::VectorXf& bias, const Eigen::VectorXf& gradients) = 0;

    // Pour les couches denses
    virtual void updateWeights(Eigen::MatrixXf& weights, const Eigen::MatrixXf& gradients) = 0;

    void setLearningRate(float lr) { learning_rate = lr; }
    float getLearningRate() const { return learning_rate; }


    void setGradientClipping(float norm) { clip_norm = norm; }

    void clipGradients(Tensor& gradients) {
        if (clip_norm <= 0.0f) return;

        // Calculer la norme L2
        float norm = 0.0f;
        for (int i = 0; i < gradients.size(); ++i) {
            norm += gradients[i] * gradients[i];
        }
        norm = std::sqrt(norm);

        // Si la norme dépasse le seuil, on scale
        if (norm > clip_norm) {
            float scale = clip_norm / norm;
            for (int i = 0; i < gradients.size(); ++i) {
                gradients[i] *= scale;
            }
        }
    }
};

class SGD : public Optimizer {
private:
    float momentum;
    Tensor velocity_w_tensor;
    Eigen::MatrixXf velocity_w_matrix;
    Eigen::VectorXf velocity_b;
    
public:
    SGD(float lr = 0.01f, float momentum = 0.9f)
        : Optimizer(lr), momentum(momentum) {}
    
    // Pour Tensor (ConvLayer)
    void updateWeights(Tensor& weights, const Tensor& gradients) override {
        if (velocity_w_tensor.size() == 0) {
            velocity_w_tensor = Tensor(weights.shape());
            velocity_w_tensor.setZero();
        }
        
        // Vérifier les dimensions
        if (velocity_w_tensor.shape() != weights.shape()) {
            velocity_w_tensor = Tensor(weights.shape());
            velocity_w_tensor.setZero();
        }
        
        #pragma omp parallel for
        for (int i = 0; i < weights.size(); ++i) {
            velocity_w_tensor[i] = momentum * velocity_w_tensor[i] - learning_rate * gradients[i];
            weights[i] += velocity_w_tensor[i];
        }
    }
    
    // Pour Eigen::MatrixXf (DenseLayer) - VERSION CORRIGÉE
    void updateWeights(Eigen::MatrixXf& weights, const Eigen::MatrixXf& gradients) override {
        // ✅ Vérification robuste des dimensions
        if (weights.rows() != gradients.rows() || weights.cols() != gradients.cols()) {
            // Essayer de transposer
            if (weights.rows() == gradients.cols() && weights.cols() == gradients.rows()) {
                std::cout << "SGD: Transposing gradients from " 
                         << gradients.rows() << "x" << gradients.cols() 
                         << " to " << gradients.cols() << "x" << gradients.rows() << std::endl;
                updateWeights(weights, gradients.transpose());
                return;
            }
            else {
                std::cerr << "❌ SGD: Dimension mismatch!" << std::endl;
                std::cerr << "  weights: " << weights.rows() << "x" << weights.cols() << std::endl;
                std::cerr << "  gradients: " << gradients.rows() << "x" << gradients.cols() << std::endl;
                throw std::runtime_error("SGD: Cannot align weights and gradients dimensions");
            }
        }
        
        // ✅ Initialisation avec les bonnes dimensions
        if (velocity_w_matrix.rows() != weights.rows() || 
            velocity_w_matrix.cols() != weights.cols()) {
            velocity_w_matrix = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
        }
        
        // ✅ Mise à jour
        velocity_w_matrix = momentum * velocity_w_matrix - learning_rate * gradients;
        weights += velocity_w_matrix;
    }
    
    void updateBias(Eigen::VectorXf& bias, const Eigen::VectorXf& gradients) override {
        // ✅ Vérification des dimensions
        if (bias.size() != gradients.size()) {
            std::cerr << "SGD: Bias dimension mismatch!" << std::endl;
            std::cerr << "  bias: " << bias.size() << std::endl;
            std::cerr << "  gradients: " << gradients.size() << std::endl;
            throw std::runtime_error("SGD: Bias size mismatch");
        }
        
        if (velocity_b.size() != bias.size()) {
            velocity_b = Eigen::VectorXf::Zero(bias.size());
        }
        
        velocity_b = momentum * velocity_b - learning_rate * gradients;
        bias += velocity_b;
    }
};
class Adam : public Optimizer {
private:
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;

    // Structure pour stocker l'état d'Adam pour chaque paramètre
    struct AdamState {
        Tensor m_tensor;
        Tensor v_tensor;
        Eigen::MatrixXf m_matrix;
        Eigen::MatrixXf v_matrix;
        Eigen::VectorXf m_b;
        Eigen::VectorXf v_b;
        int t = 0;  // Compteur INDIVIDUEL pour chaque paramètre
    };

    // Map pour associer un état à chaque pointeur de poids
    std::unordered_map<void*, AdamState> states;

public:
    Adam(float lr = 0.001f) : Optimizer(lr) {}

    void updateWeights(Tensor& weights, const Tensor& gradients) override {
        Tensor grads = gradients;
        clipGradients(grads);
        // Obtenir ou créer l'état pour ces poids
        AdamState& state = states[&weights];
        state.t++;

        if (state.m_tensor.size() == 0) {
            state.m_tensor = Tensor(weights.shape());
            state.v_tensor = Tensor(weights.shape());
            state.m_tensor.setZero();
            state.v_tensor.setZero();
        }

        // Vérifier les tailles
        if (state.m_tensor.size() != weights.size()) {
            state.m_tensor = Tensor(weights.shape());
            state.v_tensor = Tensor(weights.shape());
            state.m_tensor.setZero();
            state.v_tensor.setZero();
        }

#pragma omp parallel for
        for (int i = 0; i < weights.size(); ++i) {
            float g = gradients[i];
            state.m_tensor[i] = beta1 * state.m_tensor[i] + (1.0f - beta1) * g;
            state.v_tensor[i] = beta2 * state.v_tensor[i] + (1.0f - beta2) * g * g;

            // Bias correction avec le t INDIVIDUEL
            float m_hat = state.m_tensor[i] / (1.0f - std::pow(beta1, state.t));
            float v_hat = state.v_tensor[i] / (1.0f - std::pow(beta2, state.t));

            weights[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }

    void updateWeights(Eigen::MatrixXf& weights, const Eigen::MatrixXf& gradients) override {
        AdamState& state = states[&weights];
        state.t++;

        if (state.m_matrix.size() == 0) {
            state.m_matrix = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
            state.v_matrix = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
        }

        // Vérifier les tailles
        if (state.m_matrix.rows() != weights.rows() || state.m_matrix.cols() != weights.cols()) {
            state.m_matrix = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
            state.v_matrix = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
        }

        state.m_matrix = beta1 * state.m_matrix + (1.0f - beta1) * gradients;
        state.v_matrix = beta2 * state.v_matrix + (1.0f - beta2) * gradients.array().square().matrix();

        Eigen::ArrayXXf m_hat = state.m_matrix.array() / (1.0f - std::pow(beta1, state.t));
        Eigen::ArrayXXf v_hat = state.v_matrix.array() / (1.0f - std::pow(beta2, state.t));

        weights.array() -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
    }

    void updateBias(Eigen::VectorXf& bias, const Eigen::VectorXf& gradients) override {
        AdamState& state = states[&bias];
        state.t++;

        if (state.m_b.size() == 0) {
            state.m_b = Eigen::VectorXf::Zero(bias.size());
            state.v_b = Eigen::VectorXf::Zero(bias.size());
        }

        // Vérifier les tailles
        if (state.m_b.size() != bias.size()) {
            state.m_b = Eigen::VectorXf::Zero(bias.size());
            state.v_b = Eigen::VectorXf::Zero(bias.size());
        }

        state.m_b = beta1 * state.m_b + (1.0f - beta1) * gradients;
        state.v_b = beta2 * state.v_b + (1.0f - beta2) * gradients.array().square().matrix();

        Eigen::ArrayXf m_hat = state.m_b.array() / (1.0f - std::pow(beta1, state.t));
        Eigen::ArrayXf v_hat = state.v_b.array() / (1.0f - std::pow(beta2, state.t));

        bias.array() -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
    }
};


class StepDecay {
private:
    float initial_lr;
    float drop_rate;
    int epochs_drop;

public:
    StepDecay(float init_lr = 0.001f, float drop = 0.5f, int step = 5)
        : initial_lr(init_lr), drop_rate(drop), epochs_drop(step) {
    }

    float getLR(int epoch) {
        return initial_lr * std::pow(drop_rate, std::floor(epoch / epochs_drop));
    }
};

