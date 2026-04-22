// ConvLayer3DModelParallel.cpp
#include "ConvLayer3DModelParallel.hpp"

ConvLayer3DModelParallel::ConvLayer3DModelParallel(int in_channels, int out_channels,
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
    buildThreadPartition();
}

void ConvLayer3DModelParallel::buildThreadPartition()
{
    int total_out = out_channels;
    int base = total_out / n_threads_;
    int rem = total_out % n_threads_;

    thread_out_offsets_.resize(n_threads_ + 1, 0);
    thread_out_counts_.resize(n_threads_, 0);
    for (int t = 0; t < n_threads_; ++t) {
        int cnt = base + (t < rem ? 1 : 0);
        thread_out_counts_[t] = cnt;
        thread_out_offsets_[t+1] = thread_out_offsets_[t] + cnt;
    }

    int cols = in_channels * kernel_d * kernel_h * kernel_w;
    Eigen::Map<Eigen::MatrixXf> full_W(weights.getData(), out_channels, cols);

    thread_weights_.clear();
    for (int t = 0; t < n_threads_; ++t) {
        int start = thread_out_offsets_[t];
        int cnt = thread_out_counts_[t];
        Eigen::Map<Eigen::MatrixXf> sub_W(full_W.data() + start * cols, cnt, cols);
        Eigen::VectorXf sub_bias = bias.segment(start, cnt);
        thread_weights_.emplace_back(ThreadWeights{std::move(sub_W), std::move(sub_bias)});
    }
}

Tensor ConvLayer3DModelParallel::forward(const Tensor& input)
{
    const int B = input.dim(0);
    const int D_in = input.dim(2);
    const int H_in = input.dim(3);
    const int W_in = input.dim(4);

    auto od = computeOutputDims(D_in, H_in, W_in);
    Tensor output(B, out_channels, od.d, od.h, od.w);

    // Sauvegarde pour backward
    input_cache_ = input;
    col_cache_ = im2col(input);   // (C_in*Kd*Kh*Kw, B * oD * oH * oW)

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < n_threads_; ++t) {
        int start = thread_out_offsets_[t];
        int cnt = thread_out_counts_[t];
        Eigen::Map<Eigen::MatrixXf> out_part(
            output.getData() + start * (od.d * od.h * od.w),
            cnt, B * od.d * od.h * od.w);
        out_part.noalias() = thread_weights_[t].W_mat * col_cache_;
        out_part.colwise() += thread_weights_[t].bias_view;
    }

    return output;
}

Tensor ConvLayer3DModelParallel::backward(const Tensor& grad_output)
{
    const Tensor& input = input_cache_;
    const int B = input.dim(0);
    const int C_in = input.dim(1);
    const int D_in = input.dim(2);
    const int H_in = input.dim(3);
    const int W_in = input.dim(4);
    auto od = computeOutputDims(D_in, H_in, W_in);

    const int num_patches = B * od.d * od.h * od.w;
    const int cols = C_in * kernel_d * kernel_h * kernel_w;

    // Gradient de sortie remis en forme (out_channels, num_patches)
    // On va calculer les gradients partiels par thread puis réduire
    Eigen::MatrixXf dW_total(out_channels, cols);
    Eigen::VectorXf db_total(out_channels);
    dW_total.setZero();
    db_total.setZero();

    Tensor grad_input(B, C_in, D_in, H_in, W_in);
    grad_input.setZero();

    #pragma omp parallel
    {
        Eigen::MatrixXf dW_local(out_channels, cols);
        Eigen::VectorXf db_local(out_channels);
        dW_local.setZero();
        db_local.setZero();

        #pragma omp for schedule(static)
        for (int t = 0; t < n_threads_; ++t) {
            int start = thread_out_offsets_[t];
            int cnt = thread_out_counts_[t];

            // Vue sur la partie correspondante de grad_output
            Eigen::Map<const Eigen::MatrixXf> dY_part(
                grad_output.getData() + start * num_patches,
                cnt, num_patches);

            // Gradient des poids : dW_local[rows] = dY_part * col_cache_.transpose()
            dW_local.middleRows(start, cnt).noalias() = dY_part * col_cache_.transpose();

            // Gradient du biais
            db_local.segment(start, cnt) = dY_part.rowwise().sum();

            // Gradient vers l'entrée : dX_part = (W_part^T) * dY_part
            Eigen::MatrixXf dX_col = thread_weights_[t].W_mat.transpose() * dY_part; // (cols, num_patches)

            // col2im pour accumuler dans grad_input
            // On utilise la fonction col2im de la classe de base (protected)
            Tensor grad_part = col2im(dX_col, D_in, H_in, W_in, B, od.d, od.h, od.w);
            // Accumulation critique (les threads écrivent dans le même tenseur)
            #pragma omp critical
            {
                for (int b = 0; b < B; ++b)
                    for (int c = 0; c < C_in; ++c)
                        for (int d = 0; d < D_in; ++d)
                            for (int h = 0; h < H_in; ++h)
                                for (int w = 0; w < W_in; ++w)
                                    grad_input(b, c, d, h, w) += grad_part(b, c, d, h, w);
            }
        }

        #pragma omp critical
        {
            dW_total += dW_local;
            db_total += db_local;
        }
    }

    // Normalisation et stockage
    dW_total /= static_cast<float>(B);
    db_total /= static_cast<float>(B * num_patches);
    convertMatrixToWeights(dW_total);
    grad_bias = db_total;

    // Libérer les caches
    col_cache_.resize(0,0);
    input_cache_ = Tensor();

    return grad_input;
}