// Data parallelism 
// Partitionne le batch sur N threads OpenMP
// Les poids sont partagés entre les threads
// Chaque thread calcule une partie du batch et met à jour les poids de manière synchr

# pragma once

#include "ConvLayer.hpp"
class ConvLayerDataParallel: public ConvLayer
{
private:
    int    n_threads_;
    Tensor input_cache_; // copie justifiée : backward relit les données via im2col_single_batch
public:
    ConvLayerDataParallel(int in_channels, int out_channels,
                         int kernel_h,    int kernel_w,
                         int stride_h = 1, int stride_w = 1,
                         int pad_h    = 0, int pad_w    = 0,
                         int n_threads = 8);

    ~ConvLayerDataParallel() = default;

    std::string getName() const override { return "ConvLayerDataParallel"; }
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;
};



