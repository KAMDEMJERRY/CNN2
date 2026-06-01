#pragma once
#include "Layer.hpp"
#include "ConvLayer3D.hpp"
#include "LayerNorm3DLayer.hpp"
#include <memory>

class PatchifyStem3D : public Layer {
public:
    PatchifyStem3D(int in_channels, int out_channels, int patch_size = 4)
        : out_channels_(out_channels)
    {
        conv_ = std::make_shared<ConvLayer3D>(
            in_channels, out_channels,
            patch_size, patch_size, patch_size,
            patch_size, patch_size, patch_size,
            0, 0, 0);
        ln_ = std::make_shared<LayerNorm3DLayer>(out_channels);
        isTrainable = true;
    }

    std::string getName() const override { return "PatchifyStem3D"; }

    int numParams() const override {
        return conv_->numParams() + ln_->numParams();
    }

    Tensor forward(const Tensor& input) override {
        return ln_->forward(conv_->forward(input));
    }

    Tensor backward(const Tensor& gradOutput) override {
        return conv_->backward(ln_->backward(gradOutput));
    }

    void updateParams(Optimizer& optimizer) override {
        conv_->updateParams(optimizer);
        ln_->updateParams(optimizer);
    }

    void saveParameters(boost::archive::binary_oarchive& archive) const override {
        conv_->saveParameters(archive);
        ln_->saveParameters(archive);
    }

    void loadParameters(boost::archive::binary_iarchive& archive) override {
        conv_->loadParameters(archive);
        ln_->loadParameters(archive);
    }

private:
    int out_channels_;
    std::shared_ptr<ConvLayer3D> conv_;
    std::shared_ptr<LayerNorm3DLayer> ln_;
};
