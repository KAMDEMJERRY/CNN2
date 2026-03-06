#pragma once
#include "Tensor.hpp"
#include <Eigen/Dense>
#include <memory>
#include <unordered_map>
#include <cmath>
#include <stdexcept>

// =============================================================================
// Utilitaires : gradient clipping
// =============================================================================
namespace GradientUtils {

inline void clipByNorm(Tensor& gradients, float max_norm) {
    if (max_norm <= 0.0f) return;

    float norm = 0.0f;
    for (int i = 0; i < gradients.size(); ++i) {
        norm += gradients[i] * gradients[i];
    }
    norm = std::sqrt(norm);

    if (norm > max_norm) {
        const float scale = max_norm / norm;
        for (int i = 0; i < gradients.size(); ++i) {
            gradients[i] *= scale;
        }
    }
}

inline void clipByNorm(Eigen::MatrixXf& gradients, float max_norm) {
    if (max_norm <= 0.0f) return;
    const float norm = gradients.norm();
    if (norm > max_norm) gradients *= (max_norm / norm);
}

inline void clipByNorm(Eigen::VectorXf& gradients, float max_norm) {
    if (max_norm <= 0.0f) return;
    const float norm = gradients.norm();
    if (norm > max_norm) gradients *= (max_norm / norm);
}

} // namespace GradientUtils


// =============================================================================
// Classe de base abstraite
// =============================================================================
class Optimizer {
protected:
    float learning_rate;
    float clip_norm = 0.0f; // 0 = désactivé

public:
    explicit Optimizer(float lr = 0.01f) : learning_rate(lr) {}
    virtual ~Optimizer() = default;

    // Couches convolutives (Tensor)
    virtual void updateWeights(Tensor& weights, const Tensor& gradients) = 0;

    // Couches denses (Eigen)
    virtual void updateWeights(Eigen::MatrixXf& weights, const Eigen::MatrixXf& gradients) = 0;
    virtual void updateBias(Eigen::VectorXf& bias, const Eigen::VectorXf& gradients) = 0;

    void  setLearningRate(float lr) { learning_rate = lr; }
    float getLearningRate() const   { return learning_rate; }
    void  setGradientClipping(float norm) { clip_norm = norm; }

protected:
    template<typename T>
    void clip(T& g) const { GradientUtils::clipByNorm(g, clip_norm); }
};


// =============================================================================
// SGD avec momentum
// =============================================================================
class SGD : public Optimizer {
public:
    explicit SGD(float lr = 0.01f, float momentum = 0.9f)
        : Optimizer(lr), momentum_(momentum) {}

    void updateWeights(Tensor& weights, const Tensor& gradients) override {
        Tensor g = gradients;
        clip(g);
        ensureShape(vel_tensor_, weights);

#pragma omp parallel for
        for (int i = 0; i < weights.size(); ++i) {
            vel_tensor_[i] = momentum_ * vel_tensor_[i] - learning_rate * g[i];
            weights[i]    += vel_tensor_[i];
        }
    }

    void updateWeights(Eigen::MatrixXf& weights, const Eigen::MatrixXf& gradients) override {
        Eigen::MatrixXf g = resolveShape(weights, gradients);
        clip(g);
        ensureShape(vel_matrix_, weights);

        vel_matrix_  = momentum_ * vel_matrix_ - learning_rate * g;
        weights     += vel_matrix_;
    }

    void updateBias(Eigen::VectorXf& bias, const Eigen::VectorXf& gradients) override {
        if (bias.size() != gradients.size()) {
            throw std::runtime_error("SGD::updateBias: taille biais/gradient incompatible ("
                + std::to_string(bias.size()) + " vs " + std::to_string(gradients.size()) + ")");
        }
        Eigen::VectorXf g = gradients;
        clip(g);
        ensureShape(vel_bias_, bias);

        vel_bias_  = momentum_ * vel_bias_ - learning_rate * g;
        bias      += vel_bias_;
    }

private:
    float            momentum_;
    Tensor           vel_tensor_;
    Eigen::MatrixXf  vel_matrix_;
    Eigen::VectorXf  vel_bias_;

    // Réinitialise la vélocité si la forme a changé
    static void ensureShape(Tensor& vel, const Tensor& ref) {
        if (vel.size() == 0 || vel.shape() != ref.shape()) {
            vel = Tensor(ref.shape());
            vel.setZero();
        }
    }
    static void ensureShape(Eigen::MatrixXf& vel, const Eigen::MatrixXf& ref) {
        if (vel.rows() != ref.rows() || vel.cols() != ref.cols()) {
            vel = Eigen::MatrixXf::Zero(ref.rows(), ref.cols());
        }
    }
    static void ensureShape(Eigen::VectorXf& vel, const Eigen::VectorXf& ref) {
        if (vel.size() != ref.size()) {
            vel = Eigen::VectorXf::Zero(ref.size());
        }
    }

    // Résout les incompatibilités de forme (transpose si possible)
    static Eigen::MatrixXf resolveShape(const Eigen::MatrixXf& weights,
                                        const Eigen::MatrixXf& gradients) {
        if (weights.rows() == gradients.rows() && weights.cols() == gradients.cols()) {
            return gradients;
        }
        if (weights.rows() == gradients.cols() && weights.cols() == gradients.rows()) {
            return gradients.transpose();
        }
        throw std::runtime_error("SGD::updateWeights: formes incompatibles ("
            + std::to_string(weights.rows()) + "x" + std::to_string(weights.cols())
            + " vs " + std::to_string(gradients.rows()) + "x" + std::to_string(gradients.cols()) + ")");
    }
};


// =============================================================================
// Adam
// =============================================================================
class Adam : public Optimizer {
public:
    explicit Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        : Optimizer(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {}

    void updateWeights(Tensor& weights, const Tensor& gradients) override {
        Tensor g = gradients;
        clip(g);

        TensorState& s = tensorStates_[&weights];
        s.t++;
        ensureShape(s.m, s.v, weights);

        const float bc1 = 1.0f - std::pow(beta1_, s.t);
        const float bc2 = 1.0f - std::pow(beta2_, s.t);

#pragma omp parallel for
        for (int i = 0; i < weights.size(); ++i) {
            s.m[i] = beta1_ * s.m[i] + (1.0f - beta1_) * g[i];
            s.v[i] = beta2_ * s.v[i] + (1.0f - beta2_) * g[i] * g[i];
            const float m_hat = s.m[i] / bc1;
            const float v_hat = s.v[i] / bc2;
            weights[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }

    void updateWeights(Eigen::MatrixXf& weights, const Eigen::MatrixXf& gradients) override {
        Eigen::MatrixXf g = gradients;
        clip(g);

        MatrixState& s = matrixStates_[&weights];
        s.t++;
        ensureShape(s.m, s.v, weights);

        s.m = beta1_ * s.m + (1.0f - beta1_) * g;
        s.v = beta2_ * s.v + (1.0f - beta2_) * g.array().square().matrix();

        const float bc1 = 1.0f - std::pow(beta1_, s.t);
        const float bc2 = 1.0f - std::pow(beta2_, s.t);

        weights.array() -= learning_rate
            * (s.m.array() / bc1)
            / (s.v.array().sqrt() / std::sqrt(bc2) + epsilon_);
    }

    void updateBias(Eigen::VectorXf& bias, const Eigen::VectorXf& gradients) override {
        Eigen::VectorXf g = gradients;
        clip(g);

        VectorState& s = vectorStates_[&bias];
        s.t++;
        ensureShape(s.m, s.v, bias);

        s.m = beta1_ * s.m + (1.0f - beta1_) * g;
        s.v = beta2_ * s.v + (1.0f - beta2_) * g.array().square().matrix();

        const float bc1 = 1.0f - std::pow(beta1_, s.t);
        const float bc2 = 1.0f - std::pow(beta2_, s.t);

        bias.array() -= learning_rate
            * (s.m.array() / bc1)
            / (s.v.array().sqrt() / std::sqrt(bc2) + epsilon_);
    }

private:
    float beta1_, beta2_, epsilon_;

    // États par pointeur de paramètre (un compteur de pas par tenseur)
    struct TensorState { Tensor m, v; int t = 0; };
    struct MatrixState { Eigen::MatrixXf m, v; int t = 0; };
    struct VectorState { Eigen::VectorXf m, v; int t = 0; };

    std::unordered_map<void*, TensorState> tensorStates_;
    std::unordered_map<void*, MatrixState> matrixStates_;
    std::unordered_map<void*, VectorState> vectorStates_;

    static void ensureShape(Tensor& m, Tensor& v, const Tensor& ref) {
        if (m.size() == 0 || m.shape() != ref.shape()) {
            m = Tensor(ref.shape()); m.setZero();
            v = Tensor(ref.shape()); v.setZero();
        }
    }
    static void ensureShape(Eigen::MatrixXf& m, Eigen::MatrixXf& v, const Eigen::MatrixXf& ref) {
        if (m.rows() != ref.rows() || m.cols() != ref.cols()) {
            m = Eigen::MatrixXf::Zero(ref.rows(), ref.cols());
            v = Eigen::MatrixXf::Zero(ref.rows(), ref.cols());
        }
    }
    static void ensureShape(Eigen::VectorXf& m, Eigen::VectorXf& v, const Eigen::VectorXf& ref) {
        if (m.size() != ref.size()) {
            m = Eigen::VectorXf::Zero(ref.size());
            v = Eigen::VectorXf::Zero(ref.size());
        }
    }
};


// =============================================================================
// Scheduler : Step Decay
// =============================================================================
class StepDecay {
public:
    StepDecay(float initial_lr = 0.001f, float drop_rate = 0.5f, int epochs_per_drop = 5)
        : initial_lr_(initial_lr), drop_rate_(drop_rate), epochs_per_drop_(epochs_per_drop) {}

    float getLR(int epoch) const {
        return initial_lr_ * std::pow(drop_rate_, std::floor(static_cast<float>(epoch) / epochs_per_drop_));
    }

    // Applique directement le decay à un optimizer
    void apply(Optimizer& optimizer, int epoch) const {
        optimizer.setLearningRate(getLR(epoch));
    }

private:
    float initial_lr_;
    float drop_rate_;
    int   epochs_per_drop_;
};