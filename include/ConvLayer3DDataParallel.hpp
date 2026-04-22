// ConvLayer3DDataParallel.hpp
#pragma once

#include "ConvLayer3D.hpp"
#include <omp.h>

class ConvLayer3DDataParallel : public ConvLayer3D {
public:
    // Constructeur identique à ConvLayer3D, avec paramètre supplémentaire n_threads
    ConvLayer3DDataParallel(int in_channels, int out_channels,
                            int kernel_d, int kernel_h, int kernel_w,
                            int stride_d = 1, int stride_h = 1, int stride_w = 1,
                            int pad_d = 0, int pad_h = 0, int pad_w = 0,
                            int n_threads = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

private:
    int n_threads_;
    Tensor input_cache_;   // copie de l'input (lecture seule en backward)

    // Fonction utilitaire pour un seul batch (remplit col)
    void im2col_single_batch_3D(const float* input_data, int batch_idx,
                                int C_in, int D_in, int H_in, int W_in,
                                int kernel_d, int kernel_h, int kernel_w,
                                int stride_d, int stride_h, int stride_w,
                                int pad_d, int pad_h, int pad_w,
                                int oD, int oH, int oW,
                                Eigen::MatrixXf& col) const;
};