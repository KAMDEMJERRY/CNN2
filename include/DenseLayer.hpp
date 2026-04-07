#pragma once
#include "Layer.hpp"
#include "ModelSerializer.hpp"
#include <Eigen/Dense>

// ─────────────────────────────────────────────────────────────────────────────
// DenseLayer — couche entièrement connectée
//
// Compatible Tensor unifié 4D et 5D :
//   Entrée  : (B, C, [D,] H, W)  — aplatie en (B, input_size) automatiquement
//   Sortie  : (B, output_size, 1, 1)  en mode 4D
//             (B, output_size, 1, 1, 1) en mode 5D
//             → compatible SoftmaxLayer dans les deux cas
// ─────────────────────────────────────────────────────────────────────────────
class DenseLayer : public Layer {
private:
    int input_size;
    int output_size;

    Eigen::MatrixXf weights;      // (output_size, input_size)
    Eigen::VectorXf bias;         // (output_size)
    Eigen::MatrixXf grad_weights; // (output_size, input_size)
    Eigen::VectorXf grad_bias;    // (output_size)

    // Cache pour le backward — conserve le rang logique de l'entrée
    Tensor          input_cache;
    int             cached_rank = 4;   // rang logique de l'entrée (4 ou 5)

public:

    DenseLayer(int input_size, int output_size);
    ~DenseLayer() override = default;

    Tensor forward (const Tensor& input)      override;
    Tensor backward(const Tensor& grad_output) override;

    void initializeWeights(const std::string& method = "xavier");

    std::string getName() const override { return "Dense"; }

    // ── Getters ───────────────────────────────────────────────────────────────
    Eigen::MatrixXf& getWeights()          { return weights;      }
    Eigen::MatrixXf& getWeightGradients()  { return grad_weights; }
    Eigen::VectorXf& getBias()             { return bias;         }
    Eigen::VectorXf& getBiasGradients()    { return grad_bias;    }
    int getInputSize()  const              { return input_size;   }
    int getOutputSize() const              { return output_size;  }

    void setWeights(const Eigen::MatrixXf& new_weights);
    void setBias   (const Eigen::VectorXf& new_bias);

    // ── updateParams — même convention que ConvLayer ──────────────────────────
    void updateParams(Optimizer& optimizer) override {
        optimizer.updateWeights(getWeights(), getWeightGradients());
        optimizer.updateBias   (getBias(),    getBiasGradients());
        grad_weights.setZero();
        grad_bias.setZero();
    }

    void saveParameters(boost::archive::binary_oarchive& archive) const override {
        archive << weights;
        archive << bias;
    }

    void loadParameters(boost::archive::binary_iarchive& archive) override {
        archive >> weights;
        archive >> bias;
    }

private:

    // Construit la sortie dans le bon rang logique
    // (B, output_size, 1, 1) en 4D  ou  (B, output_size, 1, 1, 1) en 5D
    Tensor buildOutput(const Eigen::MatrixXf& matrix, int batch_size) const {
        if (cached_rank == 4)
            return Tensor::fromMatrix(matrix, {batch_size, output_size, 1, 1});
        else
            return Tensor::fromMatrix(matrix, {batch_size, output_size, 1, 1, 1});
    }

    // Reconstruit le gradient d'entrée dans le rang et la shape d'origine
    Tensor buildGradInput(const Eigen::MatrixXf& grad_matrix) const {
        return Tensor::fromMatrix(grad_matrix, input_cache.shape());
    }
};