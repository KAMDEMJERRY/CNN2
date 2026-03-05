
# include "CNNBuilder.hpp"

int main() {

    // DimensionCalculator::printArchitecture(28, 28, 1);
    // ImageNet: 224x224x3
    std::cout << "\n=== MNIST Architecture ===" << std::endl;
    CNNBuilder mnist(28, 28, 3);

    mnist.addConvSame(32, 3);      // 28x28x32
    mnist.addConvSame(32, 3);      // 28x28x32
    mnist.addMaxPool(2);           // 14x14x32

    mnist.addConvSame(64, 3);      // 14x14x64
    mnist.addConvSame(64, 3);      // 14x14x64
    mnist.addMaxPool(2);           // 7x7x64

    mnist.addFlatten();            // 3136
    mnist.addDense(128);
    mnist.addDense(10);

    mnist.printArchitecture();
    std::cout << "Flatten size: " << mnist.getFirstDenseInputSize() << std::endl;
    return 0;
}
