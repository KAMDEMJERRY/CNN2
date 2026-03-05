#pragma once
#include "Layer.hpp"
#include <algorithm>
#include <cmath>

class MaxPoolLayer : public Layer {
private:
    int pool_size;
    int stride;

    // Cache pour le backward pass
    Tensor input_cache;
    std::vector<std::vector<int>> max_indices; // [batch][position] -> index dans l'input

public:

    MaxPoolLayer(int pool_size, int stride) : pool_size(pool_size), stride(stride) {}

    ~MaxPoolLayer() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    

    std::string getName() const override { return "MaxPool"; }

private:
    // Calcul des dimensions de sortie
    std::pair<int, int> computeOutputDims(int height, int width) const;

    // Structure pour stocker les indices des maximums
    struct MaxInfo {
        float value;
        int batch_idx;
        int channel_idx;
        int input_h;
        int input_w;
    };
};