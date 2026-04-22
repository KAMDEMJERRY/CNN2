// ConvLayer3DModelParallel.hpp
#pragma once

#include "ConvLayer3D.hpp"
#include <omp.h>
#include <vector>

class ConvLayer3DModelParallel : public ConvLayer3D {
public:
    ConvLayer3DModelParallel(int in_channels, int out_channels,
                             int kernel_d, int kernel_h, int kernel_w,
                             int stride_d = 1, int stride_h = 1, int stride_w = 1,
                             int pad_d = 0, int pad_h = 0, int pad_w = 0,
                             int n_threads = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

    std::string getName() const override { return "ConvLayer3DModelParallel"; }

private:
    int n_threads_;
    std::vector<int> thread_out_offsets_;
    std::vector<int> thread_out_counts_;
    struct ThreadWeights {
        Eigen::Map<Eigen::MatrixXf> W_mat;
        Eigen::VectorXf bias_view;
    };
    std::vector<ThreadWeights> thread_weights_;
    Tensor input_cache_;          // sauvegarde de l'input pour le backward
    Eigen::MatrixXf col_cache_;   // sauvegarde de im2col pour éviter recalcul

    void buildThreadPartition();
};