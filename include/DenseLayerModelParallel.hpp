// Model parallelism for DenseLayer
//
// Principe : les neurones de sortie (D_out) sont partitionnés entre N threads.
// Chaque thread est propriétaire d'une tranche exclusive de la matrice des poids W,
// ce qui élimine tout besoin de section critique dans le backward.
//
// Comparaison avec DenseLayerDataParallel :
//   DataParallel  → partition sur le batch (B)  — les poids sont partagés (read-only)
//   ModelParallel → partition sur D_out          — chaque thread a sa tranche de W

#pragma once

#include "DenseLayer.hpp"
#include <omp.h>

class DenseLayerModelParallel : public DenseLayer {
private:
    int n_threads_;
    int chunk_size_;   // nombre de neurones de sortie par thread (= ceil(D_out / n_threads))

public:
    DenseLayerModelParallel(int input_size, int output_size, int n_threads = 8);
    ~DenseLayerModelParallel() = default;

    std::string getName() const override { return "DenseModelParallel"; }

    Tensor forward (const Tensor& input)      override;
    Tensor backward(const Tensor& gradOutput) override;
};
