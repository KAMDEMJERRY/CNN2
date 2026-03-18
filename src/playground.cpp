#include "Tensor.hpp"
#include "SparseTensor.hpp"
#include "SparseConvLayer3D.hpp"
#include <iostream>
#include <fstream>



int main() {

    int B = 1;
    int C = 1;
    int D = 28;
    int H = 28;
    int W = 28;
    float val = 3.4;
    Tensor input(B, C, D, H, W);
    input.setConstant(val);
    // Chargement : Tensor dense → SparseTensor
    SparseTensor sp = SparseTensor::from_dense(input, 1e-4f);

    // Bloc sparse
    SparseConvLayer3D conv1(1, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1, true); // SubManifold
    sp = conv1.forward(sp);
    sp.applyReLU();

    SparseConvLayer3D conv2(16, 32, 3, 3, 3, 2, 2, 2, 1, 1, 1, false); // stride=2
    sp = conv2.forward(sp);
    sp.applyReLU();
    
    // Retour vers dense pour DenseLayer
    Tensor pooled = sp.globalAvgPool(); // (B, 32, 1,1,1)
    // → DenseLayer(32, 256) comme avant

    return 0;
}