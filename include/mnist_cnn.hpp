# pragma once
# include "ConvLayer.hpp"
# include "ActivationLayer.hpp"
# include "MaxPoolLayer.hpp"
# include "DenseLayer.hpp"
# include "DropoutLayer.hpp"
# include "LossLayer.hpp"
# include "Optimizer.hpp"
# include "Dimensions.hpp"
# include "CNN.hpp"

// Architecture qui fonctionne pour MNIST
CNN mnist_cnn() {
    CNN model;
    
    // Feature extraction
    model.addLayer(std::make_unique<ConvLayer>(1, 32, 3, 3, 1, 1, 1, 1));
    model.addLayer(std::make_unique<ReLULayer>());
    model.addLayer(std::make_unique<ConvLayer>(32, 32, 3, 3, 1, 1, 1, 1));
    model.addLayer(std::make_unique<ReLULayer>());
    model.addLayer(std::make_unique<MaxPoolLayer>(2, 2));  // 14x14
    
    model.addLayer(std::make_unique<ConvLayer>(32, 64, 3, 3, 1, 1, 1, 1));
    model.addLayer(std::make_unique<ReLULayer>());
    model.addLayer(std::make_unique<ConvLayer>(64, 64, 3, 3, 1, 1, 1, 1));
    model.addLayer(std::make_unique<ReLULayer>());
    model.addLayer(std::make_unique<MaxPoolLayer>(2, 2));  // 7x7
    
    // Classification
    model.addLayer(std::make_unique<DenseLayer>(7*7*64, 128));
    model.addLayer(std::make_unique<ReLULayer>());
    model.addLayer(std::make_unique<DropoutLayer>(0.5f));
    model.addLayer(std::make_unique<DenseLayer>(128, 10));
    model.addLayer(std::make_unique<SoftmaxLayer>());
    
    return model;
}