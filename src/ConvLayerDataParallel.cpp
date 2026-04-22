#include "ConvLayerDataParallel.hpp"
#include <omp.h>

// =============================================================================
// Constructeur
// =============================================================================

ConvLayerDataParallel::ConvLayerDataParallel(int in_channels, int out_channels,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int n_threads)
    : ConvLayer(in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w),
    n_threads_(n_threads > 0 ? n_threads : omp_get_max_threads())
{
    omp_set_num_threads(n_threads_);    
}

// =============================================================================
// im2col pour un seul batch element
// Remplit col [C_in*kH*kW, oH*oW] en accédant directement au buffer de input.
// Pas d'extraction de batch : on travaille avec un offset calculé sur getData().
// =============================================================================

static void im2col_single_batch(
    const float* input_data,   // getData() du tenseur complet
    int batch_idx,
    int C_in, int H_in, int W_in,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    int oH,       int oW,
    Eigen::MatrixXf& col)
{
    // Strides en mémoire RowMajor pour un Tensor 4D stocké comme (B,C,1,H,W)
    // Layout interne : dimension 5D -> (B, C, 1, H, W)
    // stride[0]=C*H*W, stride[1]=H*W, stride[2]=H*W (D=1), stride[3]=W, stride[4]=1
    const int stride_b  = C_in * H_in * W_in;
    const int stride_c  = H_in * W_in;
    const int stride_h_ = W_in;
    // const int stride_w_ = 1;

    const float* batch_ptr = input_data + batch_idx * stride_b;

    for (int ic = 0; ic < C_in; ++ic) {
        const float* chan_ptr = batch_ptr + ic * stride_c;

        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int row = (ic * kernel_h + kh) * kernel_w + kw;

                for (int oh = 0; oh < oH; ++oh) {
                    const int ih = oh * stride_h + kh - pad_h;

                    if (ih >= 0 && ih < H_in) {
                        const float* row_ptr = chan_ptr + ih * stride_h_;

                        for (int ow = 0; ow < oW; ++ow) {
                            const int iw = ow * stride_w + kw - pad_w;
                            col(row, oh * oW + ow) = (iw >= 0 && iw < W_in)
                                ? row_ptr[iw]
                                : 0.0f;
                        }
                    } else {
                        // Ligne hors-limites : zéro-padding
                        for (int ow = 0; ow < oW; ++ow)
                            col(row, oh * oW + ow) = 0.0f;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Forward — data parallelism : chaque thread traite un sous-ensemble du batch
//
// Optimisations zero-copy :
//  - input_cache_ptr_ : pointeur, pas de copie du tenseur d'entrée
//  - W_mat            : Eigen::Map sur le buffer des poids (weightsToMatrix())
//  - col_local        : pré-allouée par thread, réutilisée à chaque itération
//  - out_map          : Map directement sur la tranche mémoire de output[b]
//                       → le GEMM écrit en place, pas de temporaire out_local
// =============================================================================

Tensor ConvLayerDataParallel::forward(const Tensor& input)
{
    const int B    = input.dim(0);
    const int C_in = input.dim(1);
    // Le tenseur 4D est stocké en 5D (B,C,1,H,W) → H est à dim(2), W à dim(3)
    const int H_in = input.dim(2);
    const int W_in = input.dim(3);
    const int C_out = out_channels_;
    auto [oH, oW] = outputDims(H_in, W_in);

    Tensor output(B, C_out, oH, oW);

    // Map sans allocation : réinterprète le buffer des poids comme matrice
    const auto W_mat = weightsToMatrix(); // Eigen::Map<const MatrixXf>

    // Copie de l'input — cohérence avec ConvLayer (durée de vie garantie)
    input_cache_ = input;

    

#pragma omp parallel
    {
        // Matrice col pré-allouée par thread (réutilisée à chaque batch element)
        Eigen::MatrixXf col_local(C_in * kernel_h_ * kernel_w_, oH * oW);

#pragma omp for schedule(static)
        for (int b = 0; b < B; ++b) {
            // im2col sans extraction de batch (accès direct au buffer)
            im2col_single_batch(
                input.getData(), b,
                C_in, H_in, W_in,
                kernel_h_, kernel_w_,
                stride_h_, stride_w_,
                pad_h_, pad_w_,
                oH, oW,
                col_local);

            // Map sur la tranche mémoire de output correspondant au batch b
            // → GEMM écrit directement dans output, sans temporaire out_local
            Eigen::Map<Eigen::MatrixXf> out_map(
                output.getData() + b * C_out * oH * oW,
                C_out, oH * oW);
            out_map.noalias() = W_mat * col_local;
            out_map.colwise() += bias_;
        }
    }

    return output;
}

// =============================================================================
// Backward — data parallelism
//
// Le backward agrège les gradients de poids sur tous les batch elements.
// Stratégie : chaque thread calcule ses gradients locaux, puis on somme.
//
// Optimisations zero-copy :
//  - input_cache_ptr_ : lecture via pointeur (pas de copie lors du forward)
//  - W_mat            : Map sur les poids
//  - dX par batch     : accumulation thread-local puis réduction atomique
// =============================================================================

Tensor ConvLayerDataParallel::backward(const Tensor& grad_output)
{
    // Récupère les dimensions depuis la copie de l'input
    const Tensor& input = input_cache_;

    const int B    = input.dim(0);
    const int H_in = input.dim(2);
    const int W_in = input.dim(3);
    const int C_in = in_channels_;
    const int C_out = out_channels_;
    auto [oH, oW] = outputDims(H_in, W_in);

    // Map sur les poids : zéro allocation
    const auto W_mat = weightsToMatrix(); // Eigen::Map<const MatrixXf>

    // Gradient de l'entrée : on accumule les contributions de chaque batch
    Tensor grad_input(B, C_in, H_in, W_in);
    grad_input.setZero();

    // Gradients de poids/biais : accumulateurs par thread puis réduction
    const int weight_cols = C_in * kernel_h_ * kernel_w_;
    Eigen::MatrixXf dW(C_out, weight_cols);
    Eigen::VectorXf db(C_out);
    dW.setZero();
    db.setZero();

   

#pragma omp parallel
    {
        // Accumulateurs locaux au thread (évite les race conditions sans mutex)
        Eigen::MatrixXf dW_local(C_out, weight_cols);
        Eigen::VectorXf db_local(C_out);
        Eigen::MatrixXf col_local(weight_cols, oH * oW);
        dW_local.setZero();
        db_local.setZero();

#pragma omp for schedule(static)
        for (int b = 0; b < B; ++b) {
            // im2col de l'input pour ce batch element
            im2col_single_batch(
                input.getData(), b,
                C_in, H_in, W_in,
                kernel_h_, kernel_w_,
                stride_h_, stride_w_,
                pad_h_, pad_w_,
                oH, oW,
                col_local);

            // Map sur la tranche de grad_output pour le batch b : [C_out, oH*oW]
            Eigen::Map<const Eigen::MatrixXf> dY_b(
                grad_output.getData() + b * C_out * oH * oW,
                C_out, oH * oW);

            // dW += dY_b × col^T
            dW_local.noalias() += dY_b * col_local.transpose();

            // db += sum over spatial
            db_local.noalias() += dY_b.rowwise().sum();

            // dX pour ce batch : W^T × dY_b  →  col2im
            Eigen::MatrixXf dX_col(weight_cols, oH * oW);
            dX_col.noalias() = W_mat.transpose() * dY_b;

            // col2im : accumulation dans grad_input[b]
            // Pas de race condition : chaque thread écrit dans un b différent
            for (int ic = 0; ic < C_in; ++ic) {
                for (int kh = 0; kh < kernel_h_; ++kh) {
                    for (int kw = 0; kw < kernel_w_; ++kw) {
                        const int row = (ic * kernel_h_ + kh) * kernel_w_ + kw;
                        for (int oh = 0; oh < oH; ++oh) {
                            const int ih = oh * stride_h_ + kh - pad_h_;
                            if (ih < 0 || ih >= H_in) continue;
                            for (int ow = 0; ow < oW; ++ow) {
                                const int iw = ow * stride_w_ + kw - pad_w_;
                                if (iw < 0 || iw >= W_in) continue;
                                grad_input(b, ic, ih, iw) += dX_col(row, oh * oW + ow);
                            }
                        }
                    }
                }
            }
        }

        // Réduction des gradients de poids/biais (section critique courte)
#pragma omp critical
        {
            dW += dW_local;
            db += db_local;
        }
    }

    // Normalise par B et stocke les gradients
    matrixToGradWeights(dW / static_cast<float>(B));
    grad_bias_ = db / static_cast<float>(B * oH * oW);

    return grad_input;
}
