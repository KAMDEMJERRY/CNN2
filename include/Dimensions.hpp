# pragma once
#include "CNN.hpp"
#include "Layer.hpp"

class DimensionCalculator {
public:
    struct ConvParams {
        int in_channels;
        int out_channels;
        int kernel_h;
        int kernel_w;
        int stride_h = 1;
        int stride_w = 1;
        int pad_h = 0;
        int pad_w = 0;
    };

    struct PoolParams {
        int pool_size;
        int stride;
    };

    struct DenseParams {
        int input_size;
        int output_size;
    };

    // Calcule la sortie d'une couche convolution
    static std::tuple<int, int, int> convOutput(int h, int w, int c, const ConvParams& p) {
        int out_h = (h + 2 * p.pad_h - p.kernel_h) / p.stride_h + 1;
        int out_w = (w + 2 * p.pad_w - p.kernel_w) / p.stride_w + 1;
        int out_c = p.out_channels;
        return { out_h, out_w, out_c };
    }

    // Calcule la sortie d'une couche pooling
    static std::tuple<int, int, int> poolOutput(int h, int w, int c, const PoolParams& p) {
        int out_h = (h - p.pool_size) / p.stride + 1;
        int out_w = (w - p.pool_size) / p.stride + 1;
        return { out_h, out_w, c };
    }

    // Calcule la taille flatten pour DenseLayer
    static int flattenSize(int h, int w, int c) {
        return h * w * c;
    }


    static void debugArchitecture(CNN& model, const Tensor& sample_input) {
        std::cout << "\n=== Debug Architecture ===" << std::endl;

        Tensor output = sample_input;
        output.printShape();

        // Stocker les couches et leurs sorties
        std::vector<std::pair<std::string, Tensor>> layer_outputs;

        for (const auto& layer : model.getLayers()) {
            output = layer->forward(output);
            layer_outputs.push_back({ layer->getName(), output });

            std::cout << layer->getName() << ": ";
            output.printShape();
        }

        // Vérifier la cohérence pour DenseLayer
        for (size_t i = 0; i < layer_outputs.size(); ++i) {
            auto& [name, tensor] = layer_outputs[i];
            if (name == "Dense") {
                // La couche précédente doit être flatten ou avoir shape [N, features, 1, 1]
                if (i > 0) {
                    auto& prev = layer_outputs[i - 1];
                    std::cout << "Dense input from " << prev.first << ": ";
                    prev.second.printShape();
                }
            }
        }
    }

    // Affiche toute l'architecture
    static void printArchitecture(int input_h, int input_w, int input_c) {
        std::cout << "\n=== Architecture CNN ===" << std::endl;
        std::cout << "Input: " << input_c << "x" << input_h << "x" << input_w << std::endl;

        int h = input_h, w = input_w, c = input_c;

        // Exemple pour l'architecture standard
        auto printLayer = [&](const std::string& name, int new_h, int new_w, int new_c) {
            std::cout << name << ": " << c << "x" << h << "x" << w
                << " → " << new_c << "x" << new_h << "x" << new_w << std::endl;
            h = new_h; w = new_w; c = new_c;
            };

        // Conv1
        auto [h1, w1, c1] = convOutput(h, w, c, { 3, 10, 3, 3, 1, 1, 1, 1 });
        printLayer("Conv1", h1, w1, c1);

        // Pool1
        auto [h2, w2, c2] = poolOutput(h, w, c, { 2, 2 });
        printLayer("Pool1", h2, w2, c2);

        // Conv2
        auto [h3, w3, c3] = convOutput(h, w, c, { 10, 8, 3, 3, 1, 1, 1, 1 });
        printLayer("Conv2", h3, w3, c3);

        // Pool2
        auto [h4, w4, c4] = poolOutput(h, w, c, { 2, 2 });
        printLayer("Pool2", h4, w4, c4);

        // Conv3
        auto [h5, w5, c5] = convOutput(h, w, c, { 8, 5, 3, 3, 1, 1, 1, 1 });
        printLayer("Conv3", h5, w5, c5);

        // Pool3
        auto [h6, w6, c6] = poolOutput(h, w, c, { 2, 2 });
        printLayer("Pool3", h6, w6, c6);

        // Flatten
        int flatten = flattenSize(h, w, c);
        std::cout << "Flatten: " << c << "x" << h << "x" << w
            << " → " << flatten << std::endl;

        // Dense1
        std::cout << "Dense1: " << flatten << " → 256" << std::endl;
        std::cout << "Dense2: " << "256 → " << "num_classes" << std::endl;
    }
};