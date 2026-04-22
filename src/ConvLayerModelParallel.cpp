#include "ConvLayerModelParallel.hpp"
#include <cassert>

ConvLayerModelParallel::ConvLayerModelParallel(int in_channels, int out_channels,
                                               int kernel_h, int kernel_w,
                                               int stride_h, int stride_w,
                                               int pad_h, int pad_w,
                                               int n_threads)
    : ConvLayer(in_channels, out_channels, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w),
      n_threads_(n_threads > 0 ? n_threads : omp_get_max_threads())
{
    omp_set_num_threads(n_threads_);
    buildThreadPartition();
}

// -----------------------------------------------------------------------------
// Partitionne les canaux de sortie entre les threads
// -----------------------------------------------------------------------------
void ConvLayerModelParallel::buildThreadPartition()
{
    int total_out = out_channels_;
    int base = total_out / n_threads_;
    int remainder = total_out % n_threads_;

    thread_out_offsets_.resize(n_threads_ + 1, 0);
    for (int t = 0; t < n_threads_; ++t) {
        int count = base + (t < remainder ? 1 : 0);
        thread_out_offsets_[t+1] = thread_out_offsets_[t] + count;
    }

    // Préparer les vues sur les poids et biais pour chaque thread
    // Les poids sont stockés dans la classe de base (Tensor weights)
    // Format matrice : (out_channels, in_channels * kernel_h * kernel_w)
    Eigen::Map<Eigen::MatrixXf> full_W(weights.getData(),
                                       out_channels_,
                                       in_channels_ * kernel_h_ * kernel_w_);
    thread_weights_.reserve(n_threads_);
    for (int t = 0; t < n_threads_; ++t) {
        int start = thread_out_offsets_[t];
        int count = thread_out_offsets_[t+1] - start;
        // Vue sur la sous-matrice des poids
        Eigen::Map<Eigen::MatrixXf> sub_W(full_W.data() + start * full_W.cols(),
                                          count, full_W.cols());
        // Vue sur le sous-vecteur du biais
        Eigen::VectorXf sub_bias = bias.segment(start, count);
        thread_weights_.emplace_back(ThreadWeights{std::move(sub_W), std::move(sub_bias)});
    }
}

// -----------------------------------------------------------------------------
// Forward : chaque thread calcule sa partie des canaux de sortie
// -----------------------------------------------------------------------------
Tensor ConvLayerModelParallel::forward(const Tensor& input)
{
    const int B = input.dim(0);
    const int C_in = input.dim(1);
    const int H_in = input.dim(2);
    const int W_in = input.dim(3);
    auto [oH, oW] = outputDims(H_in, W_in);

    Tensor output(B, out_channels_, oH, oW);
    input_cache_ = input;  // pour le backward

    // Pré-calcul im2col sur tout le batch (partagé entre threads)
    // On utilise la méthode im2col de la classe de base (protected)
    // Note : im2col retourne une matrice (C_in*kH*kW, B*oH*oW)
    Eigen::MatrixXf col_matrix = im2col(input);

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < n_threads_; ++t) {
        int start = thread_out_offsets_[t];
        int count = thread_out_offsets_[t+1] - start;

        // Map sur la tranche de output correspondant à ce thread
        Eigen::Map<Eigen::MatrixXf> out_part(
            output.getData() + start * oH * oW,
            count, B * oH * oW);

        // GEMM partiel : (count, B*oH*oW) = (count, C_in*kH*kW) * (C_in*kH*kW, B*oH*oW)
        out_part.noalias() = thread_weights_[t].W_mat * col_matrix;
        out_part.colwise() += thread_weights_[t].bias_view;
    }

    return output;
}

// -----------------------------------------------------------------------------
// Backward : les gradients sont partitionnés de manière symétrique
// -----------------------------------------------------------------------------
Tensor ConvLayerModelParallel::backward(const Tensor& grad_output)
{
    const Tensor& input = input_cache_;
    const int B = input.dim(0);
    const int H_in = input.dim(2);
    const int W_in = input.dim(3);
    const int C_in = in_channels_;
    auto [oH, oW] = outputDims(H_in, W_in);

    // im2col sur l'entrée (partagé)
    Eigen::MatrixXf col_matrix = im2col(input);

    // Gradients totaux (initialisés à zéro)
    Eigen::MatrixXf dW_total(out_channels_, in_channels_ * kernel_h_ * kernel_w_);
    Eigen::VectorXf db_total(out_channels_);
    dW_total.setZero();
    db_total.setZero();

    Tensor grad_input(B, C_in, H_in, W_in);
    grad_input.setZero();

    #pragma omp parallel
    {
        // Accumulateurs locaux par thread
        Eigen::MatrixXf dW_local(out_channels_, in_channels_ * kernel_h_ * kernel_w_);
        Eigen::VectorXf db_local(out_channels_);
        dW_local.setZero();
        db_local.setZero();

        #pragma omp for schedule(static)
        for (int t = 0; t < n_threads_; ++t) {
            int start = thread_out_offsets_[t];
            int count = thread_out_offsets_[t+1] - start;

            // Vue sur la partie de grad_output correspondant à ce thread
            Eigen::Map<const Eigen::MatrixXf> dY_part(
                grad_output.getData() + start * oH * oW,
                count, B * oH * oW);

            // Gradient local des poids : dW_local[thread] = dY_part * col_matrix^T
            // On ne remplit que les lignes concernées (start à start+count)
            dW_local.middleRows(start, count).noalias() = dY_part * col_matrix.transpose();

            // Gradient local du biais
            db_local.segment(start, count) = dY_part.rowwise().sum();

            // Gradient vers l'entrée : dX = (W^T) * dY
            // On utilise la sous-matrice des poids transposée
            Eigen::MatrixXf dX_part = thread_weights_[t].W_mat.transpose() * dY_part;

            // Col2im partiel : accumulation dans grad_input
            // Même logique que ConvLayer::col2im mais avec dX_part au lieu de dX_col complet
            // On réutilise la fonction col2im de la classe de base (protected)
            Tensor grad_input_part = col2im(dX_part, H_in, W_in, B, oH, oW);
            // Accumulation dans grad_input (attention : les threads écrivent dans le même tenseur)
            #pragma omp critical
            {
                for (int b = 0; b < B; ++b)
                    for (int c = 0; c < C_in; ++c)
                        for (int h = 0; h < H_in; ++h)
                            for (int w = 0; w < W_in; ++w)
                                grad_input(b, c, h, w) += grad_input_part(b, c, h, w);
            }
        }

        // Réduction des gradients de poids/biais
        #pragma omp critical
        {
            dW_total += dW_local;
            db_total += db_local;
        }
    }

    // Normalisation et stockage (identique à ConvLayer)
    dW_total /= static_cast<float>(B);
    db_total /= static_cast<float>(B * oH * oW);
    matrixToGradWeights(dW_total);
    grad_bias_ = db_total;

    return grad_input;
}

// Note : La classe de base ConvLayer doit rendre protected :
// - im2col(const Tensor&) -> Eigen::MatrixXf
// - col2im(const Eigen::MatrixXf&, int H_in, int W_in, int B, int oH, int oW) -> Tensor
// - matrixToGradWeights(Eigen::MatrixXf&)
// - bias_, grad_bias_, weights (via getters)