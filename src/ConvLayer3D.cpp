#include "ConvLayer3D.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// Constructeur
// ─────────────────────────────────────────────────────────────────────────────
ConvLayer3D::ConvLayer3D(int in_channels, int out_channels,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w)
    : in_channels(in_channels), out_channels(out_channels)
    , kernel_d(kernel_d), kernel_h(kernel_h), kernel_w(kernel_w)
    , stride_d(stride_d), stride_h(stride_h), stride_w(stride_w)
    , pad_d(pad_d), pad_h(pad_h), pad_w(pad_w)
    , weights(out_channels, in_channels, kernel_d, kernel_h, kernel_w)
    , grad_weights(out_channels, in_channels, kernel_d, kernel_h, kernel_w)
    , bias(Eigen::VectorXf(out_channels))
    , grad_bias(Eigen::VectorXf(out_channels))
{
    isTrainable = true;
    initializeWeights("he");
}

// ─────────────────────────────────────────────────────────────────────────────
// initializeWeights
// Calquée sur ConvLayer : random_device, même formules He / Xavier
// fan_in étendu à la 3ème dimension du kernel
// ─────────────────────────────────────────────────────────────────────────────
void ConvLayer3D::initializeWeights(const std::string& method) {
    std::random_device rd;
    std::mt19937 gen(rd());

    float scale;
    if (method == "he") {
        // He initialization pour ReLU — fan_in inclut la profondeur du kernel
        scale = std::sqrt(2.0f / (in_channels * kernel_d * kernel_h * kernel_w));
    }
    else {
        // Xavier / Glorot
        scale = std::sqrt(1.0f / (in_channels * kernel_d * kernel_h * kernel_w));
    }

    std::normal_distribution<float> dist(0.0f, scale);

    for (int i = 0; i < weights.size(); ++i)
        weights[i] = dist(gen);

    bias.setZero();
    grad_weights.setZero();
    grad_bias.setZero();
}

void ConvLayer3D::setBias(const Eigen::VectorXf& new_bias) {
    if (new_bias.size() != out_channels)
        throw std::runtime_error("[ConvLayer3D] setBias: taille incorrecte");
    bias = new_bias;
}

// ─────────────────────────────────────────────────────────────────────────────
// convertWeightsToMatrix
// Tensor (out_ch, in_ch, Kd, Kh, Kw) → MatrixXf (out_ch, in_ch×Kd×Kh×Kw)
// Extension directe de ConvLayer::convertWeightsToMatrix
// ─────────────────────────────────────────────────────────────────────────────
void ConvLayer3D::convertWeightsToMatrix(Eigen::MatrixXf& W) const {
    int cols = in_channels * kernel_d * kernel_h * kernel_w;
    W.resize(out_channels, cols);

#pragma omp parallel for collapse(2)
    for (int oc = 0; oc < out_channels; ++oc)
        for (int ic = 0; ic < in_channels; ++ic)
            for (int kd = 0; kd < kernel_d; ++kd)
                for (int kh = 0; kh < kernel_h; ++kh)
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int col_idx = ic * (kernel_d * kernel_h * kernel_w)
                            + kd * (kernel_h * kernel_w)
                            + kh * kernel_w
                            + kw;
                        W(oc, col_idx) = weights(oc, ic, kd, kh, kw);
                    }
}

Eigen::Map<const Eigen::MatrixXf> ConvLayer3D::weightsToMatrix() const {
    int rows = out_channels;
    int cols = in_channels * kernel_d * kernel_h * kernel_w;
    return Eigen::Map<const Eigen::MatrixXf>(
        weights.getData(),
        rows,
        cols
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// convertMatrixToWeights
// MatrixXf (out_ch, in_ch×Kd×Kh×Kw) → Tensor grad_weights
// Même validation de dimensions que ConvLayer
// ─────────────────────────────────────────────────────────────────────────────
void ConvLayer3D::convertMatrixToWeights(Eigen::MatrixXf& dW) {
    int expected_cols = in_channels * kernel_d * kernel_h * kernel_w;
    if (dW.rows() != out_channels || dW.cols() != expected_cols)
        throw std::runtime_error("[ConvLayer3D] convertMatrixToWeights: dimension mismatch");

    grad_weights = Tensor(out_channels, in_channels, kernel_d, kernel_h, kernel_w);

#pragma omp parallel for collapse(2)
    for (int oc = 0; oc < out_channels; ++oc)
        for (int ic = 0; ic < in_channels; ++ic)
            for (int kd = 0; kd < kernel_d; ++kd)
                for (int kh = 0; kh < kernel_h; ++kh)
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int col_idx = ic * (kernel_d * kernel_h * kernel_w)
                            + kd * (kernel_h * kernel_w)
                            + kh * kernel_w
                            + kw;
                        grad_weights(oc, ic, kd, kh, kw) = dW(oc, col_idx);
                    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Im2Col 3D
//
// Extension directe de ConvLayer::im2col :
//   même structure de boucles, même calcul de patch_idx et row_idx,
//   avec la dimension profondeur (kd, id) ajoutée.
//
// Entrée  : (B, C_in, D_in, H_in, W_in)
// Sortie  : (C_in×Kd×Kh×Kw,  B×D_out×H_out×W_out)
// ─────────────────────────────────────────────────────────────────────────────
Eigen::MatrixXf ConvLayer3D::im2col(const Tensor& input) const {
    int B = input.dim(0);
    int C = input.dim(1);
    int D_in = input.dim(2);
    int H_in = input.dim(3);
    int W_in = input.dim(4);

    auto od = computeOutputDims(D_in, H_in, W_in);

    int patch_size = C * kernel_d * kernel_h * kernel_w;
    int num_patch = B * od.d * od.h * od.w;

    Eigen::MatrixXf col_matrix(patch_size, num_patch);

#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b)
        for (int od_ = 0; od_ < od.d; ++od_)
            for (int oh = 0; oh < od.h; ++oh)
                for (int ow = 0; ow < od.w; ++ow) {
                    // Index de colonne — même logique que ConvLayer étendue à D_out
                    int patch_idx = b * (od.d * od.h * od.w)
                        + od_ * (od.h * od.w)
                        + oh * od.w
                        + ow;

                    for (int c = 0; c < C; ++c)
                        for (int kd = 0; kd < kernel_d; ++kd)
                            for (int kh = 0; kh < kernel_h; ++kh)
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    int id = od_ * stride_d + kd - pad_d;
                                    int ih = oh * stride_h + kh - pad_h;
                                    int iw = ow * stride_w + kw - pad_w;

                                    // row_idx — même formule que ConvLayer, étendue à Kd
                                    int row_idx = c * (kernel_d * kernel_h * kernel_w)
                                        + kd * (kernel_h * kernel_w)
                                        + kh * kernel_w
                                        + kw;

                                    if (id >= 0 && id < D_in &&
                                        ih >= 0 && ih < H_in &&
                                        iw >= 0 && iw < W_in) {
                                        col_matrix(row_idx, patch_idx) = input(b, c, id, ih, iw);
                                    }
                                    else {
                                        col_matrix(row_idx, patch_idx) = 0.0f;  // zero padding
                                    }
                                }
                }

    return col_matrix;
}

// ─────────────────────────────────────────────────────────────────────────────
// Col2Im 3D
//
// Extension directe de ConvLayer::col2im :
//   même accumulation +=, mêmes calculs ih/iw étendus à id.
//
// ─────────────────────────────────────────────────────────────────────────────
Tensor ConvLayer3D::col2im(const Eigen::MatrixXf& col_matrix,
    int in_d, int in_h, int in_w) const {
    int B = input_dims_cache_.b;
    auto od = computeOutputDims(in_d, in_h, in_w);

    Tensor grad_input(B, in_channels, in_d, in_h, in_w);
    grad_input.setZero();

    for (int b = 0; b < B; ++b)
        for (int od_ = 0; od_ < od.d; ++od_)
            for (int oh = 0; oh < od.h; ++oh)
                for (int ow = 0; ow < od.w; ++ow) {
                    int col_idx = b * (od.d * od.h * od.w)
                        + od_ * (od.h * od.w)
                        + oh * od.w
                        + ow;

                    for (int ic = 0; ic < in_channels; ++ic)
                        for (int kd = 0; kd < kernel_d; ++kd)
                            for (int kh = 0; kh < kernel_h; ++kh)
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    int id = od_ * stride_d + kd - pad_d;
                                    int ih = oh * stride_h + kh - pad_h;
                                    int iw = ow * stride_w + kw - pad_w;

                                    if (id >= 0 && id < in_d &&
                                        ih >= 0 && ih < in_h &&
                                        iw >= 0 && iw < in_w) {
                                        int row_idx = ic * (kernel_d * kernel_h * kernel_w)
                                            + kd * (kernel_h * kernel_w)
                                            + kh * kernel_w
                                            + kw;
                                        // Accumulation : même convention que ConvLayer
                                        grad_input(b, ic, id, ih, iw) += col_matrix(row_idx, col_idx);
                                    }
                                }
                }

    return grad_input;
}

// ─────────────────────────────────────────────────────────────────────────────
// gradOutputToMatrix
// (B, C_out, D_out, H_out, W_out) → (C_out, B×D_out×H_out×W_out)
// Extension de ConvLayer::gradOutputToMatrix avec la dimension D_out
// ─────────────────────────────────────────────────────────────────────────────
Eigen::MatrixXf ConvLayer3D::gradOutputToMatrix(const Tensor& grad_output,
    const OutDims& od) const {
    int B = grad_output.dim(0);
    Eigen::MatrixXf dO(out_channels, B * od.d * od.h * od.w);

#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; ++b)
        for (int od_ = 0; od_ < od.d; ++od_)
            for (int oh = 0; oh < od.h; ++oh)
                for (int ow = 0; ow < od.w; ++ow) {
                    int col_idx = b * (od.d * od.h * od.w)
                        + od_ * (od.h * od.w)
                        + oh * od.w
                        + ow;
                    for (int oc = 0; oc < out_channels; ++oc)
                        dO(oc, col_idx) = grad_output(b, oc, od_, oh, ow);
                }

    return dO;
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward
//
// Même séquence que ConvLayer::forward :
//   1. im2col        → col_cache
//   2. weights → W_mat
//   3. GEMM : W_mat × col_cache
//   4. bias (colwise)
//   5. reshape → Tensor (B, C_out, D_out, H_out, W_out)
// ─────────────────────────────────────────────────────────────────────────────
Tensor ConvLayer3D::forward(const Tensor& input) {
    if (input.dim(1) != in_channels)
        throw std::runtime_error("[ConvLayer3D] forward: in_channels mismatch");

    // Stocke uniquement les dimensions — les données ne sont pas relues en backward
    input_dims_cache_ = { input.dim(0), input.dim(2), input.dim(3), input.dim(4) };

    int B    = input.dim(0);
    int D_in = input.dim(2);
    int H_in = input.dim(3);
    int W_in = input.dim(4);
    
    

    auto od = computeOutputDims(D_in, H_in, W_in);

    // 1. Poids en matrice
    // Eigen::MatrixXf W_mat;
    // convertWeightsToMatrix(W_mat);

    // 2. Im2Col
    col_cache = im2col(input);


    // 3. GEMM : (C_out, C_in×Kd×Kh×Kw) × (C_in×Kd×Kh×Kw, B×D_out×H_out×W_out)
    // Eigen::MatrixXf out_mat = W_mat * col_cache;

    // 4. Biais — identique à ConvLayer : colwise()
    // out_mat.colwise() += bias;

    // 3. Reshape → Tensor (B, C_out, D_out, H_out, W_out)
    Tensor output(B, out_channels, od.d, od.h, od.w);

    // #pragma omp parallel for collapse(4)
    //     for (int b   = 0; b   < B;           ++b)
    //     for (int oc  = 0; oc  < out_channels; ++oc)
    //     for (int od_ = 0; od_ < od.d;         ++od_)
    //     for (int oh  = 0; oh  < od.h;         ++oh)
    //     for (int ow  = 0; ow  < od.w;         ++ow) {
    //         int col_idx = b  * (od.d * od.h * od.w)
    //                     + od_ * (od.h * od.w)
    //                     + oh  *  od.w
    //                     + ow;
    //         output(b, oc, od_, oh, ow) = out_mat(oc, col_idx);
    //     }


    // GEMM  ecrit directement dans le buffer ouput via Map pas de buffer temporaire
    Eigen::Map<Eigen::MatrixXf> out_map(output.getData(), out_channels, B * od.d * od.h * od.w);
    out_map.noalias() = weightsToMatrix() * col_cache;
    out_map.colwise() += bias;

    return output;
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward
//
// Même séquence que ConvLayer::backward :
//   1. gradOutputToMatrix → dO
//   2. dW = dO × col^T  /  batch_size          (même normalisation)
//   3. db = dO.rowwise().sum() / (batch × Dout×Hout×Wout)
//   4. dCol = W^T × dO
//   5. col2im → grad_input
//   6. vider col_cache
// ─────────────────────────────────────────────────────────────────────────────
Tensor ConvLayer3D::backward(const Tensor& gradOutput) {
    int B    = input_dims_cache_.b;
    int D_in = input_dims_cache_.d;
    int H_in = input_dims_cache_.h;
    int W_in = input_dims_cache_.w;
    auto od = computeOutputDims(D_in, H_in, W_in);
    
    // 1. Reshape gradOutput
    Eigen::MatrixXf dO = gradOutputToMatrix(gradOutput, od);

    
    // 2. Gradient des poids — même normalisation que ConvLayer : / batch_size
    Eigen::MatrixXf dW = dO * col_cache.transpose();
    dW /= static_cast<float>(B);
    convertMatrixToWeights(dW);

    // 3. Gradient du biais — même normalisation : / (batch × D_out × H_out × W_out)
    grad_bias = dO.rowwise().sum();
    grad_bias /= static_cast<float>(B * od.d * od.h * od.w);

    // 4. Gradient vers l'entrée — weightsToMatrix() = Map, zéro copie
    Eigen::MatrixXf dCol = weightsToMatrix().transpose() * dO;

    // 5. Col2Im
    Tensor grad_input = col2im(dCol, D_in, H_in, W_in);

    // 6. Libérer le cache — même convention que ConvLayer
    col_cache = Eigen::MatrixXf();

    return grad_input;
}