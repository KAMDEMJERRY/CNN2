#pragma once
#include "SparseTensor.hpp"
#include "Optimizer.hpp"
#include "ModelSerializer.hpp"
#include "Tensor.hpp"
#include <random>

// ---------------------------------------------------------------------------
// DepthwiseSparseConv3D
//
// Convolution sparse depthwise spécifique aux blocs ConvNeXt.
// Ne supporte QUE le mode SubManifold (entrée_coords == sortie_coords)
// et stride = 1.
// ---------------------------------------------------------------------------
class DepthwiseSparseConv3D {
public:
    DepthwiseSparseConv3D(int channels, int kernel_size = 7)
        : channels_(channels), k_(kernel_size), pad_(kernel_size / 2)
    {
        weights_ = Tensor(channels, 1, k_, k_, k_);
        grad_w_  = Tensor(channels, 1, k_, k_, k_);
        
        bias_    = Eigen::VectorXf::Zero(channels);
        grad_b_  = Eigen::VectorXf::Zero(channels);

        initializeWeights();
    }

    int numParams() const { return channels_ * k_ * k_ * k_ + channels_; }

    void initializeWeights() {
        const int fan_in = k_ * k_ * k_;
        const float scale = std::sqrt(2.0f / fan_in);
        std::mt19937 gen{std::random_device{}()};
        std::normal_distribution<float> dist(0.0f, scale);

        for (int i = 0; i < weights_.size(); ++i) {
            weights_[i] = dist(gen);
        }
    }

    SparseTensor forward(SparseTensor& input) { // we pass non-const so we can buildLookup() if needed
        input.buildLookup(); // Ensure lookup is ready
        
        input_cache_ = input;
        
        SparseTensor output = input; // Copy coords
        output.features.setZero();   // but reset features
        
        const int N = input.nnz();

        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            const int b = input.coords(i, 0);
            const int d = input.coords(i, 1);
            const int h = input.coords(i, 2);
            const int w = input.coords(i, 3);
            
            for (int kd = 0; kd < k_; ++kd) {
                for (int kh = 0; kh < k_; ++kh) {
                    for (int kw = 0; kw < k_; ++kw) {
                        int id = d - pad_ + kd;
                        int ih = h - pad_ + kh;
                        int iw = w - pad_ + kw;
                        
                        if (id >= 0 && id < input.spatial_d &&
                            ih >= 0 && ih < input.spatial_h &&
                            iw >= 0 && iw < input.spatial_w) 
                        {
                            int idx = input.find(b, id, ih, iw);
                            if (idx >= 0) {
                                for(int c = 0; c < channels_; ++c) {
                                    output.features(i, c) += input.features(idx, c) * weights_(c, 0, kd, kh, kw);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Add bias
        output.features.rowwise() += bias_.transpose();

        return output;
    }

    SparseTensor backward(const SparseTensor& gradOutput) {
        SparseTensor gradInput = gradOutput;
        gradInput.features.setZero();

        grad_w_.setZero();
        grad_b_.setZero();

        const int N = gradOutput.nnz();
        const int B = gradOutput.batch_size;

        // OMP locks or critical sections would be needed for gradInput and grad_w.
        // For simplicity and avoiding race conditions, we iterate without OMP here for gradients.
        // It's a research codebase, safety > speed for backward pass on small custom layers.

        for (int i = 0; i < N; ++i) {
            const int b = gradOutput.coords(i, 0);
            const int d = gradOutput.coords(i, 1);
            const int h = gradOutput.coords(i, 2);
            const int w = gradOutput.coords(i, 3);
            
            for (int c = 0; c < channels_; ++c) {
                grad_b_[c] += gradOutput.features(i, c);
            }

            for (int kd = 0; kd < k_; ++kd) {
                for (int kh = 0; kh < k_; ++kh) {
                    for (int kw = 0; kw < k_; ++kw) {
                        int id = d - pad_ + kd;
                        int ih = h - pad_ + kh;
                        int iw = w - pad_ + kw;

                        if (id >= 0 && id < input_cache_.spatial_d &&
                            ih >= 0 && ih < input_cache_.spatial_h &&
                            iw >= 0 && iw < input_cache_.spatial_w) 
                        {
                            int idx = input_cache_.find(b, id, ih, iw);
                            if (idx >= 0) {
                                for(int c = 0; c < channels_; ++c) {
                                    float go = gradOutput.features(i, c);
                                    gradInput.features(idx, c) += go * weights_(c, 0, kd, kh, kw);
                                    grad_w_(c, 0, kd, kh, kw) += go * input_cache_.features(idx, c);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Normalise bias and weights by batch size
        grad_b_ /= static_cast<float>(B);
        for(int i = 0; i < grad_w_.size(); ++i) grad_w_[i] /= static_cast<float>(B);

        return gradInput;
    }

    void updateParams(Optimizer& optimizer) {
        optimizer.updateWeights(weights_, grad_w_);
        optimizer.updateBias(bias_, grad_b_);
    }

    void saveParameters(boost::archive::binary_oarchive& archive) const {
        archive << weights_ << bias_;
    }

    void loadParameters(boost::archive::binary_iarchive& archive) {
        archive >> weights_ >> bias_;
    }

private:
    int channels_;
    int k_;
    int pad_;

    Tensor weights_;
    Tensor grad_w_;
    Eigen::VectorXf bias_;
    Eigen::VectorXf grad_b_;

    SparseTensor input_cache_;
};
