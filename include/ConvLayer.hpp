#pragma once
#include "Layer.hpp"
#include "ModelSerializer.hpp"

// =============================================================================
// ConvLayer — convolution 2D avec im2col + GEMM Eigen
// Poids : [out_channels, in_channels, kernel_h, kernel_w]
// =============================================================================
class ConvLayer : public Layer {
public:

    ConvLayer(int in_channels, int out_channels,
              int kernel_h,    int kernel_w,
              int stride_h = 1, int stride_w = 1,
              int pad_h    = 0, int pad_w    = 0);

    ~ConvLayer() override = default;

    // Non-copiable (cache interne non partageable)
    ConvLayer(const ConvLayer&)            = delete;
    ConvLayer& operator=(const ConvLayer&) = delete;

    // --- Layer interface ---
    Tensor      forward (const Tensor& input)     override;
    Tensor      backward(const Tensor& gradOutput) override;
    std::string getName()                    const override { return "ConvLayer"; }

    void updateParams(Optimizer& optimizer) override {
        optimizer.updateWeights(weights_,     grad_weights_);
        optimizer.updateBias   (bias_,        grad_bias_);
        grad_weights_.setZero();
        grad_bias_.setZero();
    }

    void saveParameters(boost::archive::binary_oarchive& archive) const override {
        archive << weights_;
        archive << bias_;
    }

    void loadParameters(boost::archive::binary_iarchive& archive) override {
        archive >> weights_;
        archive >> bias_;
    }

    // --- Initialisation ---
    // method : "he" (ReLU) ou "xavier"
    void initializeWeights(const std::string& method = "he");

    // --- Accesseurs ---
    Tensor&          getWeights()          { return weights_;      }
    Tensor&          getWeightGradients()  { return grad_weights_; }
    Eigen::VectorXf& getBias()             { return bias_;         }
    Eigen::VectorXf& getBiasGradients()    { return grad_bias_;    }

    void setWeights(const Tensor& w)          { weights_ = w;    }
    void setBias   (const Eigen::VectorXf& b) { bias_    = b;    }

protected:
    // --- Hyperparamètres ---
    const int in_channels_;
    const int out_channels_;
    const int kernel_h_, kernel_w_;
    const int stride_h_, stride_w_;
    const int pad_h_,    pad_w_;

    // --- Paramètres apprenables ---
    Tensor          weights_;       // [out_ch, in_ch, kH, kW]
    Eigen::VectorXf bias_;          // [out_ch]

    // --- Gradients ---
    Tensor          grad_weights_;
    Eigen::VectorXf grad_bias_;

    // --- Cache (forward → backward) ---
    struct InputDims { int b, h, w; };  // seules les dims sont nécessaires en backward
    InputDims       input_dims_cache_{};
    Eigen::MatrixXf col_cache_;         // im2col de l'entrée

    // --- Helpers : dimensions ---
    std::pair<int,int> outputDims(int in_h, int in_w) const {
        return { (in_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1,
                 (in_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1 };
    }

    // Indice linéaire dans le patch : (ic, kh, kw) → ligne de la col-matrix
    int patchRow(int ic, int kh, int kw) const {
        return ic * kernel_h_ * kernel_w_ + kh * kernel_w_ + kw;
    }

    // --- im2col / col2im ---
    Eigen::MatrixXf im2col(const Tensor& input) const;
    Tensor          col2im(const Eigen::MatrixXf& col,
                           int height, int width) const;

    // --- Conversion poids ↔ matrice ---
    // Retourne un Map [out_ch, in_ch*kH*kW] directement sur le buffer weights_ (sans copie)
    Eigen::Map<const Eigen::MatrixXf> weightsToMatrix() const;

    // Remplit grad_weights_ depuis une matrice [out_ch, in_ch*kH*kW]
    void matrixToGradWeights(const Eigen::MatrixXf& m);

    // --- grad_output → matrice [out_ch, B*oH*oW] ---
    Eigen::MatrixXf gradOutputToMatrix(const Tensor& grad_output,
                                       int out_h, int out_w) const;
};