#pragma once
#include "ConvLayer3D.hpp"
#include <Eigen/Dense>

// ---------------------------------------------------------------------------
// DepthwiseConvLayer3D
//
// Convolution 3D groupée où groups = in_channels = out_channels.
// Chaque canal est convolué indépendamment avec son propre filtre 3D.
// ---------------------------------------------------------------------------
class DepthwiseConvLayer3D : public Layer {
public:
    DepthwiseConvLayer3D(int channels, int kernel_d, int kernel_h, int kernel_w,
                         int stride_d = 1, int stride_h = 1, int stride_w = 1,
                         int pad_d = 0, int pad_h = 0, int pad_w = 0)
        : channels_(channels), kd_(kernel_d), kh_(kernel_h), kw_(kernel_w),
          sd_(stride_d), sh_(stride_h), sw_(stride_w),
          pd_(pad_d), ph_(pad_h), pw_(pad_w)
    {
        // weights : [channels, 1, Kd, Kh, Kw]
        weights_    = Tensor(channels, 1, kd_, kh_, kw_);
        grad_w_     = Tensor(channels, 1, kd_, kh_, kw_);
        bias_       = Eigen::VectorXf::Zero(channels);
        grad_b_     = Eigen::VectorXf::Zero(channels);
        isTrainable = true;

        initializeWeights();
    }

    std::string getName() const override { return "DepthwiseConv3D"; }

    int numParams() const override { return channels_ * kd_ * kh_ * kw_ + channels_; }

    void initializeWeights() {
        const int fan_in = kd_ * kh_ * kw_;
        const float scale = std::sqrt(2.0f / fan_in); // He init for depthwise
        std::mt19937 gen{std::random_device{}()};
        std::normal_distribution<float> dist(0.0f, scale);

        for (int i = 0; i < weights_.size(); ++i) {
            weights_[i] = dist(gen);
        }
    }

    Tensor forward(const Tensor& input) override {
        if (input.ndim() != 5 || input.dim(1) != channels_)
            throw std::invalid_argument("[DepthwiseConvLayer3D] Invaild input dims");

        const int B = input.dim(0);
        const int in_d = input.dim(2);
        const int in_h = input.dim(3);
        const int in_w = input.dim(4);

        const int out_d = (in_d + 2 * pd_ - kd_) / sd_ + 1;
        const int out_h = (in_h + 2 * ph_ - kh_) / sh_ + 1;
        const int out_w = (in_w + 2 * pw_ - kw_) / sw_ + 1;

        Tensor output(B, channels_, out_d, out_h, out_w);
        input_cache_ = input;
        
        // Boucle simple for research-level code. Unfold pour chaque canal par itération
        #pragma omp parallel for
        for (int c = 0; c < channels_; ++c) {
            for (int b = 0; b < B; ++b) {
                for (int od = 0; od < out_d; ++od) {
                    for (int oh = 0; oh < out_h; ++oh) {
                        for (int ow = 0; ow < out_w; ++ow) {
                            float sum = 0.0f;
                            for (int kd = 0; kd < kd_; ++kd) {
                                for (int kh = 0; kh < kh_; ++kh) {
                                    for (int kw = 0; kw < kw_; ++kw) {
                                        int id = od * sd_ - pd_ + kd;
                                        int ih = oh * sh_ - ph_ + kh;
                                        int iw = ow * sw_ - pw_ + kw;

                                        if (id >= 0 && id < in_d && ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                            sum += input(b, c, id, ih, iw) * weights_(c, 0, kd, kh, kw);
                                        }
                                    }
                                }
                            }
                            output(b, c, od, oh, ow) = sum + bias_[c];
                        }
                    }
                }
            }
        }

        return output;
    }

    Tensor backward(const Tensor& gradOutput) override {
        const int B = gradOutput.dim(0);
        const int in_d = input_cache_.dim(2);
        const int in_h = input_cache_.dim(3);
        const int in_w = input_cache_.dim(4);
        
        const int out_d = gradOutput.dim(2);
        const int out_h = gradOutput.dim(3);
        const int out_w = gradOutput.dim(4);

        Tensor gradInput(input_cache_.shape());
        gradInput.setZero();
        grad_w_.setZero();
        grad_b_.setZero();

        // 1. Gradients w.r.t Bias and Weights
        for (int c = 0; c < channels_; ++c) {
            float b_grad = 0.0f;
            for (int b = 0; b < B; ++b) {
                for (int od = 0; od < out_d; ++od) {
                    for (int oh = 0; oh < out_h; ++oh) {
                        for (int ow = 0; ow < out_w; ++ow) {
                            const float go = gradOutput(b, c, od, oh, ow);
                            b_grad += go;

                            for (int kd = 0; kd < kd_; ++kd) {
                                for (int kh = 0; kh < kh_; ++kh) {
                                    for (int kw = 0; kw < kw_; ++kw) {
                                        int id = od * sd_ - pd_ + kd;
                                        int ih = oh * sh_ - ph_ + kh;
                                        int iw = ow * sw_ - pw_ + kw;
                                        if (id >= 0 && id < in_d && ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                            grad_w_(c, 0, kd, kh, kw) += go * input_cache_(b, c, id, ih, iw);
                                            gradInput(b, c, id, ih, iw) += go * weights_(c, 0, kd, kh, kw);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            grad_b_[c] = b_grad / static_cast<float>(B);
        }

        // Normalisation par le batch size pour grad_weights
        for(int i = 0; i < grad_w_.size(); ++i) {
            grad_w_[i] /= static_cast<float>(B);
        }

        return gradInput;
    }

    void updateParams(Optimizer& optimizer) override {
        optimizer.updateWeights(weights_, grad_w_);
        optimizer.updateBias(bias_, grad_b_);
    }

    void saveParameters(boost::archive::binary_oarchive& archive) const override {
        archive << weights_ << bias_;
    }

    void loadParameters(boost::archive::binary_iarchive& archive) override {
        archive >> weights_ >> bias_;
    }

private:
    int channels_;
    int kd_, kh_, kw_;
    int sd_, sh_, sw_;
    int pd_, ph_, pw_;

    Tensor weights_;
    Tensor grad_w_;
    Eigen::VectorXf bias_;
    Eigen::VectorXf grad_b_;
    
    Tensor input_cache_;
};
