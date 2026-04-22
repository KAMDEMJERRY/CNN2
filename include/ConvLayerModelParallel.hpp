#pragma once

#include "ConvLayer.hpp"
#include <omp.h>
#include <vector>

/**
 * @brief Model parallelism pour une couche de convolution 2D.
 * 
 * Partitionne les filtres (canaux de sortie) entre plusieurs threads.
 * Chaque thread calcule une partie de la sortie sur l'intégralité du batch.
 * Les résultats sont concaténés selon la dimension des canaux de sortie.
 */
class ConvLayerModelParallel : public ConvLayer {
public:
    /**
     * @param in_channels  Nombre de canaux d'entrée
     * @param out_channels Nombre de canaux de sortie
     * @param kernel_h     Hauteur du noyau
     * @param kernel_w     Largeur du noyau
     * @param stride_h     Pas vertical
     * @param stride_w     Pas horizontal
     * @param pad_h        Padding vertical
     * @param pad_w        Padding horizontal
     * @param n_threads    Nombre de threads OpenMP (0 = max)
     */
    ConvLayerModelParallel(int in_channels, int out_channels,
                           int kernel_h, int kernel_w,
                           int stride_h = 1, int stride_w = 1,
                           int pad_h = 0, int pad_w = 0,
                           int n_threads = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

    std::string getName() const override { return "ConvLayerModelParallel"; }

private:
    int n_threads_;
    int out_channels_per_thread_;   // Nombre de filtres par thread (dernier peut être plus petit)
    std::vector<int> thread_out_offsets_; // Offset dans les canaux de sortie pour chaque thread

    // Sous-ensembles de poids/biais pour chaque thread (vues sans copie)
    struct ThreadWeights {
        Eigen::Map<Eigen::MatrixXf> W_mat; // Vue sur une sous-matrice des poids
        Eigen::VectorXf bias_view;         // Vue sur un sous-vecteur du biais
    };
    std::vector<ThreadWeights> thread_weights_;

    void buildThreadPartition();
};