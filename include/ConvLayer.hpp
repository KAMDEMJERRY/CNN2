# pragma once
# include "Layer.hpp"


class ConvLayer : public Layer {
private:
    int in_channels;
    int out_channels;
    int kernel_h;
    int kernel_w;
    int stride;
    int pad_h;
    int pad_w;

    // Poids: [out_channels, in_channels, kernel_h, kernel_w]
    Tensor weights;

    // Biais: [out_channels]
    Eigen::VectorXf bias;

    Tensor grad_weights;

    Eigen::VectorXf grad_bias;

    // Cache pour la backward pass
    Tensor input_cache;

    Eigen::MatrixXf col_cache;

    // Methode im2col optimisee avec Eigen
    Eigen::MatrixXf im2col(const Tensor& input) const;

    // Methode col2im optimisee avec Eigen
    Tensor col2im(const Eigen::MatrixXf& col_matrix, int height, int width) const;

    Eigen::MatrixXf gradOutputToMatrix(const Tensor& grad_output, int out_height, int out_width) const;


public:

    // ConvLayer(int in_channels, int out_channels,
    //     int kernel_size, int stride = 1,
    //     int padding = 0);

    ConvLayer(int in_channels, int out_channels,
        int kernel_h, int kernel_w,
        int stride_h = 1, int stride_w = 1,
        int pad_h = 0, int pad_w = 0);

    ~ConvLayer() override = default;

    ConvLayer(const ConvLayer&) = delete;
    ConvLayer& operator=(const ConvLayer&) = delete;


    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;


    void initializeWeights(const std::string& method = "he"); // "xavier/glorot";
    void setWeights(const Tensor& new_weights){weights = new_weights;};
    void setBias(const Eigen::VectorXf& new_bias);

    std::string getName() const override { return "Convlution"; }

    // Getters for weights and bias
    Tensor& getWeights() { return weights; }
    Tensor& getWeightGradients() { return grad_weights; }
    Eigen::VectorXf& getBias() { return bias; }
    Eigen::VectorXf& getBiasGradients() { return grad_bias; }

    void updateParams(Optimizer& optimizer) override {
        optimizer.updateWeights(getWeights(), getWeightGradients());
        optimizer.updateBias(getBias(), getBiasGradients());
        grad_weights.setZero();
        grad_bias.setZero();
    }
    
private:
    // Helper function to calculate output dimensions
    std::pair<int, int> computeOutputDims(int input_h, int input_w) const {
        int out_h = (input_h + 2 * pad_h - kernel_h) / stride + 1;
        int out_w = (input_w + 2 * pad_w - kernel_w) / stride + 1;
        return { out_h, out_w };
    }

    void convertWeightsToMatrix(Eigen::MatrixXf& weights_matrix) const {
        // Reorganiser les poids
# pragma omp parallel for collapse(2)
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int col_idx = ic * kernel_h * kernel_w + kh * kernel_w + kw;
                        weights_matrix(oc, col_idx) = weights(oc, ic, kh, kw);
                    }
                }
            }
        }
    }

    void convertMatrixToWeights(Eigen::MatrixXf& grad_weights_matrix) {
        // Vérifier que la matrice a les bonnes dimensions
        if (grad_weights_matrix.rows() != out_channels ||
            grad_weights_matrix.cols() != in_channels * kernel_h * kernel_w) {
            throw std::runtime_error("Matrix dimensions don't match ConvLayer parameters");
        }

        // Initialiser grad_weights avec les bonnes dimensions (au lieu de reshape)
        grad_weights = Tensor(out_channels, in_channels, kernel_h, kernel_w);


        // Reorganiser les poids
# pragma omp parallel for collapse(2)
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int col_idx = ic * kernel_h * kernel_w + kh * kernel_w + kw;
                        grad_weights(oc, ic, kh, kw) = grad_weights_matrix(oc, col_idx);
                    }
                }
            }
        }
    }


};