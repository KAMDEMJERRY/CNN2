#pragma once
#include "Layer.hpp"
#include "ConvLayer3D.hpp"
#include "LayerNorm3DLayer.hpp"
#include <memory>

class DenseDownsample3D : public Layer {
public:
    DenseDownsample3D(int in_channels, int out_channels)
        : in_channels_(in_channels), out_channels_(out_channels)
    {
        ln_ = std::make_shared<LayerNorm3DLayer>(in_channels);
        conv_ = std::make_shared<ConvLayer3D>(
            in_channels, out_channels,
            2, 2, 2, // kernel
            2, 2, 2, // stride
            0, 0, 0);// pad
        isTrainable = true;
    }

    std::string getName() const override { return "DenseDownsample3D"; }

    int numParams() const override {
        return ln_->numParams() + conv_->numParams();
    }

    Tensor forward(const Tensor& input) override {
        return conv_->forward(ln_->forward(input));
    }

    Tensor backward(const Tensor& gradOutput) override {
        return ln_->backward(conv_->backward(gradOutput));
    }

    void updateParams(Optimizer& optimizer) override {
        ln_->updateParams(optimizer);
        conv_->updateParams(optimizer);
    }

    void saveParameters(boost::archive::binary_oarchive& archive) const override {
        ln_->saveParameters(archive);
        conv_->saveParameters(archive);
    }

    void loadParameters(boost::archive::binary_iarchive& archive) override {
        ln_->loadParameters(archive);
        conv_->loadParameters(archive);
    }

private:
    int in_channels_, out_channels_;
    std::shared_ptr<LayerNorm3DLayer> ln_;
    std::shared_ptr<ConvLayer3D> conv_;
};
