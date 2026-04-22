#include "ConvLayer3DDataParallel.hpp"
#include <omp.h>

ConvLayer3DDataParallel::ConvLayer3DDataParallel(int in_channels, int out_channels,
                                                 int kernel_d, int kernel_h, int kernel_w,
                                                 int stride_d, int stride_h, int stride_w,
                                                 int pad_d, int pad_h, int pad_w,
                                                 int n_threads)
    : ConvLayer3D(in_channels, out_channels,
                  kernel_d, kernel_h, kernel_w,
                  stride_d, stride_h, stride_w,
                  pad_d, pad_h, pad_w),
      n_threads_(n_threads > 0 ? n_threads : omp_get_max_threads())
{
    omp_set_num_threads(n_threads_);
}

// -----------------------------------------------------------------------------
// im2col pour un seul élément du batch
// -----------------------------------------------------------------------------
void ConvLayer3DDataParallel::im2col_single_batch_3D(
    const float* input_data, int batch_idx,
    int C_in, int D_in, int H_in, int W_in,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int oD, int oH, int oW,
    Eigen::MatrixXf& col) const
{
    const int stride_b = C_in * D_in * H_in * W_in;
    const int stride_c = D_in * H_in * W_in;
    const int stride_d_ = H_in * W_in;
    const int stride_h_ = W_in;

    const float* batch_ptr = input_data + batch_idx * stride_b;
    const int patch_size = C_in * kernel_d * kernel_h * kernel_w;
    const int num_patches = oD * oH * oW;

    for (int ic = 0; ic < C_in; ++ic) {
        const float* chan_ptr = batch_ptr + ic * stride_c;
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int row = ic * (kernel_d * kernel_h * kernel_w)
                            + kd * (kernel_h * kernel_w)
                            + kh * kernel_w
                            + kw;
                    for (int od = 0; od < oD; ++od) {
                        int id = od * stride_d + kd - pad_d;
                        bool id_ok = (id >= 0 && id < D_in);
                        if (id_ok) {
                            const float* depth_ptr = chan_ptr + id * stride_d_;
                            for (int oh = 0; oh < oH; ++oh) {
                                int ih = oh * stride_h + kh - pad_h;
                                bool ih_ok = (ih >= 0 && ih < H_in);
                                if (ih_ok) {
                                    const float* row_ptr = depth_ptr + ih * stride_h_;
                                    for (int ow = 0; ow < oW; ++ow) {
                                        int iw = ow * stride_w + kw - pad_w;
                                        int col_idx = od * (oH * oW) + oh * oW + ow;
                                        col(row, col_idx) = (iw >= 0 && iw < W_in) ? row_ptr[iw] : 0.0f;
                                    }
                                } else {
                                    for (int ow = 0; ow < oW; ++ow) {
                                        int col_idx = od * (oH * oW) + oh * oW + ow;
                                        col(row, col_idx) = 0.0f;
                                    }
                                }
                            }
                        } else {
                            for (int oh = 0; oh < oH; ++oh) {
                                for (int ow = 0; ow < oW; ++ow) {
                                    int col_idx = od * (oH * oW) + oh * oW + ow;
                                    col(row, col_idx) = 0.0f;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Forward
// -----------------------------------------------------------------------------
Tensor ConvLayer3DDataParallel::forward(const Tensor& input)
{
    int B    = input.dim(0);
    int C_in = input.dim(1);
    int D_in = input.dim(2);
    int H_in = input.dim(3);
    int W_in = input.dim(4);

    auto od = computeOutputDims(D_in, H_in, W_in);
    Tensor output(B, out_channels, od.d, od.h, od.w);

    input_cache_ = input;   // copie pour backward

    auto W_mat = weightsToMatrix();
    const int patch_size = C_in * kernel_d * kernel_h * kernel_w;
    const int num_patches = od.d * od.h * od.w;

#pragma omp parallel
    {
        Eigen::MatrixXf col_local(patch_size, num_patches);
#pragma omp for schedule(static)
        for (int b = 0; b < B; ++b) {
            im2col_single_batch_3D(
                input.getData(), b,
                C_in, D_in, H_in, W_in,
                kernel_d, kernel_h, kernel_w,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                od.d, od.h, od.w,
                col_local);

            Eigen::Map<Eigen::MatrixXf> out_map(
                output.getData() + b * out_channels * num_patches,
                out_channels, num_patches);
            out_map.noalias() = W_mat * col_local;
            out_map.colwise() += bias;
        }
    }
    return output;
}

// -----------------------------------------------------------------------------
// Backward
// -----------------------------------------------------------------------------
Tensor ConvLayer3DDataParallel::backward(const Tensor& grad_output)
{
    const Tensor& input = input_cache_;
    int B    = input.dim(0);
    int C_in = input.dim(1);
    int D_in = input.dim(2);
    int H_in = input.dim(3);
    int W_in = input.dim(4);

    auto od = computeOutputDims(D_in, H_in, W_in);
    const int patch_size = C_in * kernel_d * kernel_h * kernel_w;
    const int num_patches = od.d * od.h * od.w;

    auto W_mat = weightsToMatrix();

    Tensor grad_input(B, C_in, D_in, H_in, W_in);
    grad_input.setZero();

    Eigen::MatrixXf dW(out_channels, patch_size);
    Eigen::VectorXf db(out_channels);
    dW.setZero();
    db.setZero();

#pragma omp parallel
    {
        Eigen::MatrixXf dW_local(out_channels, patch_size);
        Eigen::VectorXf db_local(out_channels);
        Eigen::MatrixXf col_local(patch_size, num_patches);
        dW_local.setZero();
        db_local.setZero();

#pragma omp for schedule(static)
        for (int b = 0; b < B; ++b) {
            im2col_single_batch_3D(
                input.getData(), b,
                C_in, D_in, H_in, W_in,
                kernel_d, kernel_h, kernel_w,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                od.d, od.h, od.w,
                col_local);

            Eigen::Map<const Eigen::MatrixXf> dY_b(
                grad_output.getData() + b * out_channels * num_patches,
                out_channels, num_patches);

            dW_local.noalias() += dY_b * col_local.transpose();
            db_local.noalias() += dY_b.rowwise().sum();

            Eigen::MatrixXf dX_col(patch_size, num_patches);
            dX_col.noalias() = W_mat.transpose() * dY_b;

            // Col2im
            for (int ic = 0; ic < C_in; ++ic) {
                for (int kd = 0; kd < kernel_d; ++kd) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int row = ic * (kernel_d * kernel_h * kernel_w)
                                    + kd * (kernel_h * kernel_w)
                                    + kh * kernel_w
                                    + kw;
                            for (int od_ = 0; od_ < od.d; ++od_) {
                                int id = od_ * stride_d + kd - pad_d;
                                if (id < 0 || id >= D_in) continue;
                                for (int oh = 0; oh < od.h; ++oh) {
                                    int ih = oh * stride_h + kh - pad_h;
                                    if (ih < 0 || ih >= H_in) continue;
                                    for (int ow = 0; ow < od.w; ++ow) {
                                        int iw = ow * stride_w + kw - pad_w;
                                        if (iw < 0 || iw >= W_in) continue;
                                        int col_idx = od_ * (od.h * od.w) + oh * od.w + ow;
                                        grad_input(b, ic, id, ih, iw) += dX_col(row, col_idx);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

#pragma omp critical
        {
            dW += dW_local;
            db += db_local;
        }
    }

    // Normalisation et stockage
    dW /= static_cast<float>(B);
    db /= static_cast<float>(B * num_patches);
    convertMatrixToWeights(dW);
    grad_bias = db;          // maintenant accessible (protected)

    return grad_input;
}