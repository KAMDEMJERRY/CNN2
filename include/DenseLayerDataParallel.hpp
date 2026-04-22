#pragma once
#include "DenseLayer.hpp"

class DenseLayerDataParallel : public DenseLayer {
public:
    // Seuils adaptatifs — justifiés par Goto & van de Geijn (2008)
    // en dessous : overhead OpenMP > gain → séquentiel
    static constexpr int THRESHOLD_B    = 64;   // batch min pour paralléliser
    static constexpr int THRESHOLD_DIM  = 256;  // dimension min pour paralléliser

    // Tailles de tuiles calibrées pour L1 cache (32KB)
    // float = 4 bytes → L1 tient 8192 floats
    // TILE_B * TILE_D * 4 bytes ≤ 32KB → 64 * 128 * 4 = 32KB
    static constexpr int TILE_B = 64;
    static constexpr int TILE_D = 128;

    explicit DenseLayerDataParallel(int input_size, int output_size,
                                    int n_threads = 0);

    Tensor forward (const Tensor& input)      override;
    Tensor backward(const Tensor& grad_output) override;

private:
    int n_threads_;

    // Stratégies forward
    Tensor forward_sequential (const Eigen::MatrixXf& X, int B);
    Tensor forward_parallel   (const Eigen::MatrixXf& X, int B);

    // Stratégies backward
    Tensor backward_sequential(const Eigen::MatrixXf& dO,
                                const Eigen::MatrixXf& X,  int B);
    Tensor backward_parallel  (const Eigen::MatrixXf& dO,
                                const Eigen::MatrixXf& X,  int B);

    // Sélecteur — cœur de la logique adaptive
    bool should_parallelize(int B, int Din, int Dout) const;
};