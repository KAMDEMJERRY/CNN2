#pragma once
#include "Layer.hpp"
#include "DepthwiseConvLayer3D.hpp"
#include "LayerNorm3DLayer.hpp"
#include "ConvLayer3D.hpp"
#include "ActivationLayer.hpp"
#include <memory>

class ConvNeXtBlock3D : public Layer {
public:
    ConvNeXtBlock3D(int channels, int kernel_size = 7, float layer_scale_init = 1e-6f)
        : channels_(channels)
    {
        int pad = (kernel_size - 1) / 2;
        dwconv_ = std::make_shared<DepthwiseConvLayer3D>(
            channels, kernel_size, kernel_size, kernel_size,
            1, 1, 1, pad, pad, pad);

        ln_ = std::make_shared<LayerNorm3DLayer>(channels);
        expand_ = std::make_shared<ConvLayer3D>(channels, 4 * channels, 1, 1, 1);
        gelu_ = std::make_shared<GELULayer>();
        contract_ = std::make_shared<ConvLayer3D>(4 * channels, channels, 1, 1, 1);

        gamma_ = Eigen::VectorXf::Constant(channels, layer_scale_init);
        grad_gamma_ = Eigen::VectorXf::Zero(channels);
        isTrainable = true;
    }

    std::string getName() const override { return "ConvNeXtBlock3D"; }

    int numParams() const override {
        return dwconv_->numParams() + ln_->numParams() + expand_->numParams() + contract_->numParams() + channels_;
    }

    Tensor forward(const Tensor& input) override {
        input_cache_ = input; // pour la skip connection backward
        
        Tensor x = dwconv_->forward(input);
        x = ln_->forward(x);
        x = expand_->forward(x);
        x = gelu_->forward(x);
        x = contract_->forward(x);

        x_pre_scale_cache_ = x; // pout le Layerscale backward

        const int B = x.dim(0);
        const int D = x.dim(2);
        const int H = x.dim(3);
        const int W = x.dim(4);

        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < channels_; ++c) {
                for (int d = 0; d < D; ++d) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            x(b, c, d, h, w) = gamma_[c] * x_pre_scale_cache_(b, c, d, h, w) + input_cache_(b, c, d, h, w);
                        }
                    }
                }
            }
        }

        return x;
    }

    Tensor backward(const Tensor& gradOutput) override {
        const int B = gradOutput.dim(0);
        const int D = gradOutput.dim(2);
        const int H = gradOutput.dim(3);
        const int W = gradOutput.dim(4);

        grad_gamma_.setZero();
        Tensor grad_contract_in(gradOutput.shape());
        Tensor grad_skip(gradOutput.shape());

        // Backward LayerScale + Skip Connection
        for (int c = 0; c < channels_; ++c) {
            float g_gamma = 0.0f;
            for (int b = 0; b < B; ++b) {
                for (int d = 0; d < D; ++d) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            const float go = gradOutput(b, c, d, h, w);
                            g_gamma += go * x_pre_scale_cache_(b, c, d, h, w);
                            grad_contract_in(b, c, d, h, w) = go * gamma_[c];
                            grad_skip(b, c, d, h, w) = go;
                        }
                    }
                }
            }
            grad_gamma_[c] = g_gamma;
        }

        Tensor g = contract_->backward(grad_contract_in);
        g = gelu_->backward(g);
        g = expand_->backward(g);
        g = ln_->backward(g);
        g = dwconv_->backward(g);

        // Add skip gradients
        for (int i = 0; i < g.size(); ++i) {
            g[i] += grad_skip[i];
        }

        return g;
    }

    void updateParams(Optimizer& optimizer) override {
        dwconv_->updateParams(optimizer);
        ln_->updateParams(optimizer);
        expand_->updateParams(optimizer);
        contract_->updateParams(optimizer);

        optimizer.updateBias(gamma_, grad_gamma_);
    }

    void saveParameters(boost::archive::binary_oarchive& archive) const override {
        dwconv_->saveParameters(archive);
        ln_->saveParameters(archive);
        expand_->saveParameters(archive);
        contract_->saveParameters(archive);
        archive << gamma_;
    }

    void loadParameters(boost::archive::binary_iarchive& archive) override {
        dwconv_->loadParameters(archive);
        ln_->loadParameters(archive);
        expand_->loadParameters(archive);
        contract_->loadParameters(archive);
        archive >> gamma_;
    }

private:
    int channels_;

    std::shared_ptr<DepthwiseConvLayer3D> dwconv_;
    std::shared_ptr<LayerNorm3DLayer> ln_;
    std::shared_ptr<ConvLayer3D> expand_;
    std::shared_ptr<GELULayer> gelu_;
    std::shared_ptr<ConvLayer3D> contract_;

    Eigen::VectorXf gamma_;
    Eigen::VectorXf grad_gamma_;

    Tensor input_cache_;
    Tensor x_pre_scale_cache_;
};
