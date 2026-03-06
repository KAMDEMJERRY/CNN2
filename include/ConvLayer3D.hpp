#pragma once
#include "Layer.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// ConvLayer3D — Convolution 3D dense sur Tensor (B, C, D, H, W)
// Calquée sur ConvLayer 2D : mêmes conventions de normalisation des gradients,
// même style im2col/col2im, même updateParams, même initializeWeights.
// ─────────────────────────────────────────────────────────────────────────────
class ConvLayer3D : public Layer {
private:
    int in_channels;
    int out_channels;
    int kernel_d, kernel_h, kernel_w;
    int stride_d, stride_h, stride_w;
    int pad_d,    pad_h,    pad_w;

    // Poids : (out_channels, in_channels, Kd, Kh, Kw)
    Tensor weights;

    // Biais : (out_channels)
    Eigen::VectorXf bias;

    // Gradients
    Tensor          grad_weights;
    Eigen::VectorXf grad_bias;

    // Cache backward
    Tensor          input_cache;
    Eigen::MatrixXf col_cache;

public:

    // ── Constructeur ──────────────────────────────────────────────────────────

    ConvLayer3D(int in_channels,  int out_channels,
                int kernel_d,     int kernel_h,   int kernel_w,
                int stride_d = 1, int stride_h = 1, int stride_w = 1,
                int pad_d    = 0, int pad_h    = 0, int pad_w    = 0);

    ~ConvLayer3D() override = default;

    ConvLayer3D(const ConvLayer3D&)            = delete;
    ConvLayer3D& operator=(const ConvLayer3D&) = delete;

    // ── Interface Layer ───────────────────────────────────────────────────────

    Tensor forward (const Tensor& input)      override;
    Tensor backward(const Tensor& gradOutput) override;
    std::string getName() const override { return "ConvLayer3D"; }

    // ── Initialisation ────────────────────────────────────────────────────────

    // "he" (défaut, pour ReLU) ou "xavier"
    // Utilise random_device comme ConvLayer — pas de seed fixe
    void initializeWeights(const std::string& method = "he");

    void setWeights(const Tensor& new_weights) { weights = new_weights; }
    void setBias   (const Eigen::VectorXf& new_bias);

    // ── Getters ───────────────────────────────────────────────────────────────

    Tensor&          getWeights()         { return weights;      }
    Tensor&          getWeightGradients() { return grad_weights; }
    Eigen::VectorXf& getBias()            { return bias;         }
    Eigen::VectorXf& getBiasGradients()   { return grad_bias;    }

    // ── updateParams — identique à ConvLayer ─────────────────────────────────

    void updateParams(Optimizer& optimizer) override {
        optimizer.updateWeights(getWeights(), getWeightGradients());
        optimizer.updateBias   (getBias(),    getBiasGradients());
        grad_weights.setZero();
        grad_bias.setZero();
    }

private:

    // ── Dimensions de sortie ──────────────────────────────────────────────────

    struct OutDims { int d, h, w; };

    OutDims computeOutputDims(int in_d, int in_h, int in_w) const {
        return {
            (in_d + 2 * pad_d - kernel_d) / stride_d + 1,
            (in_h + 2 * pad_h - kernel_h) / stride_h + 1,
            (in_w + 2 * pad_w - kernel_w) / stride_w + 1
        };
    }

    // ── Im2Col 3D ─────────────────────────────────────────────────────────────
    // (B, C_in, D, H, W) → (C_in×Kd×Kh×Kw,  B×D_out×H_out×W_out)
    // Même structure de boucles que ConvLayer::im2col, étendue à la profondeur
    Eigen::MatrixXf im2col(const Tensor& input) const;

    // ── Col2Im 3D ─────────────────────────────────────────────────────────────
    // Inverse d'im2col — accumulation += sur (D, H, W)
    // Même signature que ConvLayer::col2im, avec in_d en plus
    Tensor col2im(const Eigen::MatrixXf& col_matrix,
                  int in_d, int in_h, int in_w) const;

    // ── gradOutputToMatrix ────────────────────────────────────────────────────
    // (B, C_out, D_out, H_out, W_out) → (C_out, B×D_out×H_out×W_out)
    Eigen::MatrixXf gradOutputToMatrix(const Tensor& grad_output,
                                       const OutDims& od) const;

    // ── Helpers poids ─────────────────────────────────────────────────────────

    // Tensor (out_ch, in_ch, Kd, Kh, Kw) → MatrixXf (out_ch, in_ch×Kd×Kh×Kw)
    void convertWeightsToMatrix(Eigen::MatrixXf& W) const;

    // MatrixXf (out_ch, in_ch×Kd×Kh×Kw) → Tensor grad_weights
    // Identique à ConvLayer::convertMatrixToWeights, étendu à Kd
    void convertMatrixToWeights(Eigen::MatrixXf& dW);
};