#include "CNN.hpp"
#include "ConvLayer.hpp"
#include "DenseLayer.hpp"
#include "Layer.hpp"
#include <iostream>
#include <cassert>
#include <memory>
#include <cmath>

int main() {
    std::cout << "=== TEST SERIALIZATION ===" << std::endl;

    CNN model1;
    model1.addLayer(std::make_shared<ConvLayer>(3, 8, 3, 3)); // in=3, out=8
    model1.addLayer(std::make_shared<DenseLayer>(8 * 26 * 26, 10)); // Just arbitrary sized dense

    // The parameters are randomly initialized in constructors
    std::string filename = "test_model_params.bin";
    
    // Save model parameters
    std::cout << "Saving model paramaters..." << std::endl;
    model1.saveParameters(filename);

    // Create a new model with exactly the same architecture
    CNN model2;
    model2.addLayer(std::make_shared<ConvLayer>(3, 8, 3, 3)); 
    model2.addLayer(std::make_shared<DenseLayer>(8 * 26 * 26, 10));

    // Load model parameters into model2
    std::cout << "Loading model paramaters..." << std::endl;
    model2.loadParameters(filename);

    // Validate parameters
    auto* conv1 = dynamic_cast<ConvLayer*>(model1.getLayer(0));
    auto* conv2 = dynamic_cast<ConvLayer*>(model2.getLayer(0));
    
    for (int i = 0; i < conv1->getWeights().size(); ++i) {
        float diff = std::abs(conv1->getWeights()[i] - conv2->getWeights()[i]);
        if (diff > 1e-6) {
            std::cerr << "Mismatch in ConvLayer Weights at index " << i << ": " 
                      << conv1->getWeights()[i] << " != " << conv2->getWeights()[i] << std::endl;
            return 1;
        }
    }

    auto* dense1 = dynamic_cast<DenseLayer*>(model1.getLayer(1));
    auto* dense2 = dynamic_cast<DenseLayer*>(model2.getLayer(1));

    for (int i = 0; i < dense1->getWeights().size(); ++i) {
        float diff = std::abs(dense1->getWeights().data()[i] - dense2->getWeights().data()[i]);
        if (diff > 1e-6) {
            std::cerr << "Mismatch in DenseLayer Weights at index " << i << ": " 
                      << dense1->getWeights().data()[i] << " != " << dense2->getWeights().data()[i] << std::endl;
            return 1;
        }
    }

    std::cout << "SUCCESS: Model parameters saved and loaded correctly!" << std::endl;
    return 0;
}
