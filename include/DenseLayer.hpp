// DenseLayer.hpp (adapté pour la nouvelle Tensor)
# pragma once
# include "Layer.hpp"
# include <Eigen/Dense>
# include <memory>

class DenseLayer : public Layer {
private:
    int input_size;
    int output_size;

    Eigen::MatrixXf weights;  // [output_size x input_size]
    Eigen::VectorXf bias;    // [output_size]

    Eigen::MatrixXf grad_weights;
    Eigen::VectorXf grad_bias;

    Tensor input_cache;

public:
    DenseLayer(int input_size, int output_size);
    ~DenseLayer() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

    void initializeWeights(const std::string& method = "xavier");

    std::string getName() const override { return "Dense"; }

    Eigen::MatrixXf& getWeights() { return weights; }
    Eigen::MatrixXf& getWeightGradients() { return grad_weights; }
    Eigen::VectorXf& getBias() { return bias; }
    Eigen::VectorXf& getBiasGradients() { return grad_bias; }

    int getInputSize() const { return input_size; }
    int getOutputSize() const { return output_size; }

    void setWeights(const Eigen::MatrixXf& new_weights);
    void setBias(const Eigen::VectorXf& new_bias);

    void updateParams(Optimizer& optimizer) override {
        optimizer.updateWeights(getWeights(), getWeightGradients());
        optimizer.updateBias(getBias(), getBiasGradients());
        grad_weights.setZero();
        grad_bias.setZero();
    }
private:

    Eigen::MatrixXf tensorToMatrix(const Tensor& tensor) const;
    Tensor matrixToTensor(const Eigen::MatrixXf& matrix, bool is_output_size) const;
};