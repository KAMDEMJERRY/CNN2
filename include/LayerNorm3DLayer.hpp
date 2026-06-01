#pragma once
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include "Layer.hpp"
#include "Tensor.hpp"
#include "Optimizer.hpp"
#include "ModelSerializer.hpp"

// ---------------------------------------------------------------------------
//  LayerNorm3DLayer — Layer Normalization pour volumes 3D denses
//
//  Entrée  : Tensor (B, C, D, H, W)
//  Sortie  : Tensor (B, C, D, H, W)
//
//  Différence avec BatchNorm3D :
//  - LayerNorm normalise "par voxel", c'est-à-dire sur la dimension C.
//  - Pas de running stats (indépendant du batch et training/eval mode)
// ---------------------------------------------------------------------------
class LayerNorm3DLayer : public Layer {
public:
    explicit LayerNorm3DLayer(int num_channels, float eps = 1e-6f)
        : num_channels_(num_channels), eps_(eps)
    {
        gamma_ = Eigen::VectorXf::Ones(num_channels);
        beta_  = Eigen::VectorXf::Zero(num_channels);
        d_gamma_ = Eigen::VectorXf::Zero(num_channels);
        d_beta_  = Eigen::VectorXf::Zero(num_channels);
        isTrainable = true;
    }

    std::string getName() const override { return "LayerNorm3D"; }

    void saveParameters(boost::archive::binary_oarchive& archive) const override {
        archive << gamma_ << beta_;
    }

    void loadParameters(boost::archive::binary_iarchive& archive) override {
        archive >> gamma_ >> beta_;
    }

    int numParams() const override { return gamma_.size() + beta_.size(); }

    Tensor forward(const Tensor& input) override {
        if (input.ndim() != 5 || input.dim(1) != num_channels_) {
            throw std::invalid_argument("[LayerNorm3D] Invalid input dimensions");
        }

        const int B = input.dim(0);
        const int C = input.dim(1);
        const int D = input.dim(2);
        const int H = input.dim(3);
        const int W = input.dim(4);

        Tensor output(input.shape());
        x_hat_.resize(B * C * D * H * W);
        inv_std_.resize(B * D * H * W);
        input_cache_ = input;

        // Normalisation par voxel, sur la dimension C
        for (int b = 0; b < B; ++b) {
            for (int d = 0; d < D; ++d) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        float sum = 0.0f;
                        float sum_sq = 0.0f;
                        for (int c = 0; c < C; ++c) {
                            const float val = input(b, c, d, h, w);
                            sum += val;
                        }
                        const float mean = sum / C;
                        
                        for (int c = 0; c < C; ++c) {
                            const float val = input(b, c, d, h, w) - mean;
                            sum_sq += val * val;
                        }
                        const float var = sum_sq / C;
                        const float inv_std = 1.0f / std::sqrt(var + eps_);

                        const int spatial_idx = ((b * D + d) * H + h) * W + w;
                        inv_std_[spatial_idx] = inv_std;

                        for (int c = 0; c < C; ++c) {
                            const float xh = (input(b, c, d, h, w) - mean) * inv_std;
                            const int full_idx = (((b * C + c) * D + d) * H + h) * W + w;
                            x_hat_[full_idx] = xh;
                            output(b, c, d, h, w) = gamma_[c] * xh + beta_[c];
                        }
                    }
                }
            }
        }

        return output;
    }

    Tensor backward(const Tensor& gradOutput) override {
        const int B = gradOutput.dim(0);
        const int C = gradOutput.dim(1);
        const int D = gradOutput.dim(2);
        const int H = gradOutput.dim(3);
        const int W = gradOutput.dim(4);

        d_gamma_.setZero();
        d_beta_.setZero();

        Tensor gradInput(gradOutput.shape());
        gradInput.setZero();

        // 1. Accumuler les gradients d_gamma et d_beta
        for (int b = 0; b < B; ++b) {
            for (int d = 0; d < D; ++d) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        for (int c = 0; c < C; ++c) {
                            const int full_idx = (((b * C + c) * D + d) * H + h) * W + w;
                            const float go = gradOutput(b, c, d, h, w);
                            d_gamma_[c] += go * x_hat_[full_idx];
                            d_beta_[c]  += go;
                        }
                    }
                }
            }
        }

        // 2. Calcul du gradient d'entrée
        const float inv_C = 1.0f / C;
        for (int b = 0; b < B; ++b) {
            for (int d = 0; d < D; ++d) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        float sum_dxhat = 0.0f;
                        float sum_dxhat_xhat = 0.0f;
                        
                        const int spatial_idx = ((b * D + d) * H + h) * W + w;
                        const float inv_s = inv_std_[spatial_idx];

                        for (int c = 0; c < C; ++c) {
                            const int full_idx = (((b * C + c) * D + d) * H + h) * W + w;
                            const float dxh = gradOutput(b, c, d, h, w) * gamma_[c];
                            sum_dxhat += dxh;
                            sum_dxhat_xhat += dxh * x_hat_[full_idx];
                        }

                        for (int c = 0; c < C; ++c) {
                            const int full_idx = (((b * C + c) * D + d) * H + h) * W + w;
                            const float dxh = gradOutput(b, c, d, h, w) * gamma_[c];
                            gradInput(b, c, d, h, w) = inv_s * (dxh - inv_C * sum_dxhat - inv_C * x_hat_[full_idx] * sum_dxhat_xhat);
                        }
                    }
                }
            }
        }

        return gradInput;
    }

    void updateParams(Optimizer& optimizer) override {
        optimizer.updateBias(gamma_, d_gamma_);
        optimizer.updateBias(beta_,  d_beta_);
        d_gamma_.setZero();
        d_beta_.setZero();
    }

    Eigen::VectorXf& getGamma() { return gamma_; }
    Eigen::VectorXf& getBeta()  { return beta_; }

private:
    int num_channels_;
    float eps_;

    Eigen::VectorXf gamma_, beta_;
    Eigen::VectorXf d_gamma_, d_beta_;

    std::vector<float> x_hat_;
    std::vector<float> inv_std_;
    Tensor input_cache_;
};
