#pragma once
#include "Layer.hpp"
#include <random>
#include <stdexcept>

class DropoutLayer : public Layer {
public:
    explicit DropoutLayer(float rate = 0.5f) : rate_(rate), scale_(1.0f / (1.0f - rate)) {
        if (rate <= 0.0f || rate >= 1.0f) {
            throw std::invalid_argument("DropoutLayer: rate doit être dans ]0, 1[");
        }
    }

    Tensor forward(const Tensor& input) override {
        if (!training_) return input;

        mask_.resize(input.size());
        Tensor output = input;

        std::bernoulli_distribution dist(1.0f - rate_);
        for (int i = 0; i < input.size(); ++i) {
            mask_[i]  = dist(rng_);
            output[i] = mask_[i] ? input[i] * scale_ : 0.0f;
        }

        return output;
    }

    Tensor backward(const Tensor& grad_output) override {
        if (!training_) return grad_output;

        if (static_cast<int>(mask_.size()) != grad_output.size()) {
            throw std::runtime_error("DropoutLayer::backward: forward() doit être appelé avant backward()");
        }

        Tensor grad_input = grad_output;
        for (int i = 0; i < grad_output.size(); ++i) {
            grad_input[i] = mask_[i] ? grad_output[i] * scale_ : 0.0f;
        }

        return grad_input;
    }

    void setTraining(bool training) { training_ = training; }
    void train() { training_ = true; }
    void eval()  { training_ = false; }

    float getRate() const { return rate_; }

    std::string getName() const override { return "DropoutLayer"; }

private:
    float rate_;
    float scale_;               // précalculé : 1 / (1 - rate)
    std::vector<bool> mask_;
    bool training_ = true;

    // RNG partagé sur la durée de vie de la couche (évite rd() à chaque forward)
    std::mt19937 rng_{ std::random_device{}() };
};