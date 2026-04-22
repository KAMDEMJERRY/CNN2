#include "DenseLayerModelParallel.hpp"
#include <omp.h>
#include <algorithm>

// =============================================================================
// Constructeur
// =============================================================================
DenseLayerModelParallel::DenseLayerModelParallel(int input_size, int output_size, int n_threads)
    : DenseLayer(input_size, output_size),
      n_threads_(n_threads > 0 ? n_threads : omp_get_max_threads()),
      chunk_size_((output_size + n_threads_ - 1) / n_threads_) // ceil division
{
    omp_set_num_threads(n_threads_);
}

// =============================================================================
// Forward — parallélisme de modèle
// =============================================================================
//
// Chaque thread t calcule les colonnes de Y correspondant à la tranche
// [j_start .. j_end[ de D_out.
//
//   Y[b, j] = X[b, :] · W[j, :]^T  +  bias[j]
//
// Pas de race condition : chaque thread écrit dans des colonnes exclusives de Y.
// =============================================================================
Tensor DenseLayerModelParallel::forward(const Tensor& input) {
    cached_rank  = input.ndim();
    input_cache  = input;

    Eigen::MatrixXf X = input.toMatrix();   // (B, D_in)
    const int B    = static_cast<int>(X.rows());
    const int Dout = output_size;
    const int Din  = input_size;

    Eigen::MatrixXf Y(B, Dout);

#pragma omp parallel for schedule(static)
    for (int j = 0; j < Dout; ++j) {
        // Y[:, j] = X · W[j, :] + bias[j]
        // W.row(j) : vecteur ligne de dimension Din
        Y.col(j) = X * weights.row(j).transpose()
                   + Eigen::VectorXf::Constant(B, bias(j));
    }

    return buildOutput(Y, B);
}

// =============================================================================
// Backward — parallélisme de modèle
// =============================================================================
//
// grad_W[j, k]  = (1/B) Σ_b dO[b,j] · X[b,k]   → parallèle sur j, sans critical
// grad_bias[j]  = (1/B) Σ_b dO[b,j]             → parallèle sur j, sans critical
//
// dX[b, k] = Σ_j dO[b,j] · W[j,k]
//   Chaque thread accumule sa contribution dans dX_local[t],
//   puis on réduit en série après la région parallèle.
// =============================================================================
Tensor DenseLayerModelParallel::backward(const Tensor& gradOutput) {
    Eigen::MatrixXf dO = gradOutput.toMatrix();  // (B, D_out)
    Eigen::MatrixXf X  = input_cache.toMatrix(); // (B, D_in)
    const int B    = static_cast<int>(dO.rows());
    const int Dout = output_size;
    const int Din  = input_size;

    // Un tableau de matrices locales pour la réduction de dX (une par thread)
    std::vector<Eigen::MatrixXf> dX_local(
        n_threads_, Eigen::MatrixXf::Zero(B, Din));

#pragma omp parallel
    {
        int tid     = omp_get_thread_num();
        int j_start = tid * chunk_size_;
        int j_end   = std::min(j_start + chunk_size_, Dout);

        // ── Gradient des poids : tranche [j_start .. j_end[ ──────────────────
        // grad_W[j, :] = (1/B) Σ_b dO[b,j] · X[b,:]
        // Pas de race : chaque thread écrit dans sa tranche exclusive de grad_weights
        for (int j = j_start; j < j_end; ++j) {
            grad_weights.row(j) = (dO.col(j).transpose() * X) / static_cast<float>(B);
        }

        // ── Gradient des biais : tranche [j_start .. j_end[ ──────────────────
        for (int j = j_start; j < j_end; ++j) {
            grad_bias(j) = dO.col(j).sum() / static_cast<float>(B);
        }

        // ── Contribution locale à dX ──────────────────────────────────────────
        // dX_local[tid] += dO[:, j_start:j_end] · W[j_start:j_end, :]
        // Chaque thread a son propre dX_local → pas de race condition.
        if (j_start < j_end) {
            dX_local[tid] +=
                dO.middleCols(j_start, j_end - j_start) *
                weights.middleRows(j_start, j_end - j_start);
        }
    } // fin région parallèle

    // ── Réduction finale de dX (séquentielle, mais légère : n_threads additions) ──
    Eigen::MatrixXf dX = Eigen::MatrixXf::Zero(B, Din);
    for (int t = 0; t < n_threads_; ++t) {
        dX += dX_local[t];
    }

    return buildGradInput(dX);
}
