# pragma once
# include "Layer.hpp"

class DropoutLayer : public Layer {
private:
    float rate;
    std::vector<bool> mask;
    bool training = true;
    
public:
    DropoutLayer(float rate = 0.5f) : rate(rate) {}
    
    Tensor forward(const Tensor& input) override {
        if (!training) return input;
        
        Tensor output = input;
        mask.resize(input.size());
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dist(1.0f - rate);
        
        for (int i = 0; i < input.size(); ++i) {
            mask[i] = dist(gen);
            output[i] = mask[i] ? input[i] / (1.0f - rate) : 0.0f;
        }
        
        return output;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_input = grad_output;
        for (int i = 0; i < grad_output.size(); ++i) {
            grad_input[i] = mask[i] ? grad_output[i] / (1.0f - rate) : 0.0f;
        }
        return grad_input;
    }


    std::string getName() const override{
        return "Dropout layer";
    }
    
    void eval() { training = false; }
    void train() { training = true; }
};