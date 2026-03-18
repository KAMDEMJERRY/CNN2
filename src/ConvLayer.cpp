#include "ConvLayer.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

// =============================================================================
// Constructeur
// =============================================================================

ConvLayer::ConvLayer(int in_channels, int out_channels,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
    : in_channels_(in_channels), out_channels_(out_channels),
    kernel_h_(kernel_h), kernel_w_(kernel_w),
    stride_h_(stride_h), stride_w_(stride_w),
    pad_h_(pad_h), pad_w_(pad_w),
    weights_(out_channels, in_channels, kernel_h, kernel_w),
    bias_(Eigen::VectorXf::Zero(out_channels)),
    grad_weights_(out_channels, in_channels, kernel_h, kernel_w),
    grad_bias_(Eigen::VectorXf::Zero(out_channels))
{
    isTrainable = true;
    initializeWeights("he");
}

// =============================================================================
// Initialisation des poids
// =============================================================================

void ConvLayer::initializeWeights(const std::string& method) {
    const int fan_in = in_channels_ * kernel_h_ * kernel_w_;
    const float scale = (method == "he")
        ? std::sqrt(2.0f / fan_in)   // He   — adapté ReLU
        : std::sqrt(1.0f / fan_in);  // Xavier — adapté tanh/sigmoïde

    std::mt19937 gen{ std::random_device{}() };
    std::normal_distribution<float> dist(0.0f, scale);

    for (int i = 0; i < weights_.size(); ++i)
        weights_[i] = dist(gen);

    bias_.setZero();
    grad_weights_.setZero();
    grad_bias_.setZero();
}

// =============================================================================
// Forward
// =============================================================================

Tensor ConvLayer::forward(const Tensor& input) {
    input_cache_ = input;

    const int B = input.dim(0);
    const int H = input.dim(2);
    const int W = input.dim(3);
    auto [oH, oW] = outputDims(H, W);

    // im2col : [in_ch*kH*kW,  B*oH*oW]
    col_cache_ = im2col(input);

    // GEMM : [out_ch, in_ch*kH*kW] × [in_ch*kH*kW, B*oH*oW]
    //      = [out_ch, B*oH*oW]
    Eigen::MatrixXf out_mat = weightsToMatrix() * col_cache_;
    out_mat.colwise() += bias_;

    // Réorganisation → Tensor [B, out_ch, oH, oW]
    Tensor output(B, out_channels_, oH, oW);

#pragma omp parallel for collapse(4)
    for (int b = 0; b < B; ++b)
        for (int oc = 0; oc < out_channels_; ++oc)
            for (int oh = 0; oh < oH; ++oh)
                for (int ow = 0; ow < oW; ++ow) {
                    const int col = b * oH * oW + oh * oW + ow;
                    output(b, oc, oh, ow) = out_mat(oc, col);
                }

    return output;
}

// =============================================================================
// Backward
// =============================================================================

Tensor ConvLayer::backward(const Tensor& grad_output) {
    const int B = input_cache_.dim(0);
    const int H = input_cache_.dim(2);
    const int W = input_cache_.dim(3);
    auto [oH, oW] = outputDims(H, W);

    // grad_output → [out_ch, B*oH*oW]
    const Eigen::MatrixXf dY = gradOutputToMatrix(grad_output, oH, oW);

    // dL/dW = dY × col^T  / B        → [out_ch, in_ch*kH*kW]
    matrixToGradWeights((dY * col_cache_.transpose()) / static_cast<float>(B));

    // dL/db = sum over spatial+batch / (B*oH*oW)
    grad_bias_ = dY.rowwise().sum() / static_cast<float>(B * oH * oW);

    // dL/dX : W^T × dY → col2im
    const Eigen::MatrixXf dX_col = weightsToMatrix().transpose() * dY;
    Tensor grad_input = col2im(dX_col, H, W);

    col_cache_ = Eigen::MatrixXf(); // libère la mémoire
    return grad_input;
}

// =============================================================================
// im2col
// =============================================================================

Eigen::MatrixXf ConvLayer::im2col(const Tensor& input) const {
    const int B = input.dim(0);
    const int H = input.dim(2);
    const int W = input.dim(3);
    auto [oH, oW] = outputDims(H, W);

    const int patch_size = in_channels_ * kernel_h_ * kernel_w_;
    const int num_patch = B * oH * oW;

    Eigen::MatrixXf col(patch_size, num_patch);

#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b)
        for (int oh = 0; oh < oH; ++oh)
            for (int ow = 0; ow < oW; ++ow) {
                const int col_idx = b * oH * oW + oh * oW + ow;

                for (int ic = 0; ic < in_channels_; ++ic)
                    for (int kh = 0; kh < kernel_h_; ++kh)
                        for (int kw = 0; kw < kernel_w_; ++kw) {
                            const int ih = oh * stride_h_ + kh - pad_h_;
                            const int iw = ow * stride_w_ + kw - pad_w_;

                            col(patchRow(ic, kh, kw), col_idx) =
                                (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                ? input(b, ic, ih, iw)
                                : 0.0f;
                        }
            }

    return col;
}

// =============================================================================
// col2im
// =============================================================================

Tensor ConvLayer::col2im(const Eigen::MatrixXf& col,
    int height, int width) const {
    const int B = input_cache_.dim(0);
    auto [oH, oW] = outputDims(height, width);

    Tensor grad_input(B, in_channels_, height, width);
    grad_input.setZero();

    // Accumulation des gradients — pas de parallélisme (race condition sur grad_input)
    for (int b = 0; b < B; ++b)
        for (int oh = 0; oh < oH; ++oh)
            for (int ow = 0; ow < oW; ++ow) {
                const int col_idx = b * oH * oW + oh * oW + ow;

                for (int ic = 0; ic < in_channels_; ++ic)
                    for (int kh = 0; kh < kernel_h_; ++kh)
                        for (int kw = 0; kw < kernel_w_; ++kw) {
                            const int ih = oh * stride_h_ + kh - pad_h_;
                            const int iw = ow * stride_w_ + kw - pad_w_;

                            if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                grad_input(b, ic, ih, iw) += col(patchRow(ic, kh, kw), col_idx);
                        }
            }

    return grad_input;
}

// =============================================================================
// Conversions poids ↔ matrice
// =============================================================================

Eigen::MatrixXf ConvLayer::weightsToMatrix() const {
    Eigen::MatrixXf m(out_channels_, in_channels_ * kernel_h_ * kernel_w_);

#pragma omp parallel for collapse(2)
    for (int oc = 0; oc < out_channels_; ++oc)
        for (int ic = 0; ic < in_channels_; ++ic)
            for (int kh = 0; kh < kernel_h_; ++kh)
                for (int kw = 0; kw < kernel_w_; ++kw)
                    m(oc, patchRow(ic, kh, kw)) = weights_(oc, ic, kh, kw);

    return m;
}

void ConvLayer::matrixToGradWeights(const Eigen::MatrixXf& m) {
    if (m.rows() != out_channels_ ||
        m.cols() != in_channels_ * kernel_h_ * kernel_w_)
        throw std::runtime_error(
            "ConvLayer::matrixToGradWeights: dimensions incompatibles");

    grad_weights_ = Tensor(out_channels_, in_channels_, kernel_h_, kernel_w_);

#pragma omp parallel for collapse(2)
    for (int oc = 0; oc < out_channels_; ++oc)
        for (int ic = 0; ic < in_channels_; ++ic)
            for (int kh = 0; kh < kernel_h_; ++kh)
                for (int kw = 0; kw < kernel_w_; ++kw)
                    grad_weights_(oc, ic, kh, kw) = m(oc, patchRow(ic, kh, kw));
}

// =============================================================================
// gradOutputToMatrix
// =============================================================================

Eigen::MatrixXf ConvLayer::gradOutputToMatrix(const Tensor& grad_output,
    int oH, int oW) const {
    const int B = grad_output.dim(0);
    Eigen::MatrixXf m(out_channels_, B * oH * oW);

#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; ++b)
        for (int oh = 0; oh < oH; ++oh)
            for (int ow = 0; ow < oW; ++ow) {
                const int col = b * oH * oW + oh * oW + ow;
                for (int oc = 0; oc < out_channels_; ++oc)
                    m(oc, col) = grad_output(b, oc, oh, ow);
            }

    return m;
}