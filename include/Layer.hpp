# pragma once
# include "Tensor.hpp"
# include "Optimizer.hpp"
# include <boost/archive/binary_oarchive.hpp>
# include <boost/archive/binary_iarchive.hpp>

# include <memory>
# include <cmath>
# include <algorithm>
# include <stdexcept>

class Layer {
public:
    bool isTrainable = false;

    virtual ~Layer() = default;
    
    // Forward pass: takes input tensor and produces output tensor
    virtual Tensor forward(const Tensor& input) = 0;

    // Backward pass: takes gradient of output and produces gradient of input
    virtual Tensor backward(const Tensor& gradOutput) = 0;

    virtual std::string getName() const = 0;


    virtual void updateParams(Optimizer& optimizer){};

    // Serialize parameters (default does nothing, overridden by layers with parameters)
    virtual void saveParameters(boost::archive::binary_oarchive& archive) const {}
    virtual void loadParameters(boost::archive::binary_iarchive& archive) {}
};