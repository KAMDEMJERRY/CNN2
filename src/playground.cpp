#include <iostream>
#include <random>
#include <map>
#include "Tensor.hpp"


using namespace std;

Tensor3D generate_random_tensor(int batch_size, int channels, int depth, int height, int width) {
    Tensor3D tensor(batch_size, channels, depth, height, width);

    // Initialisation du générateur aléatoire
    static default_random_engine generator;
    static normal_distribution<float> distribution(0.0f, 1.0f);

    // Remplir le tensor avec des valeurs aléatoires
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int d = 0; d < depth; d++) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        tensor(b, c, d, h, w) = distribution(generator);
                    }
                }
            }
        }
    }

    return tensor;
}


int main() {

    Tensor3D ten1 =  generate_random_tensor(1, 3, 2, 2, 2);
    Tensor3D ten2 =  generate_random_tensor(1, 3, 2, 2, 2);


    Eigen::MatrixXf mat1 = ten1.toMatrix();
    Eigen::MatrixXf mat2 = ten2.toMatrix();
    Eigen::MatrixXf sum = ten2.toMatrix();


    sum = mat1 + mat2;

    std::cout << mat1 << "\n\n";
    std::cout << mat2 << "\n\n";
    std::cout << sum << "\n\n";

    Tensor3D sumT = Tensor3D::fromMatrix(sum, ten1.shape());

    ten1.printByChannel();
    ten2.printByChannel();
    sumT.printByChannel();

    return 0;
}