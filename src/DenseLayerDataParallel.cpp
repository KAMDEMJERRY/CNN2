#include "DenseLayerDataParallel.hpp"
#include <omp.h>
#include <algorithm>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// Constructeur
// ─────────────────────────────────────────────────────────────────────────────
DenseLayerDataParallel::DenseLayerDataParallel(int input_size,
                                               int output_size,
                                               int n_threads)
    : DenseLayer(input_size, output_size),
      n_threads_(n_threads > 0 ? n_threads : omp_get_max_threads())
{}

// ─────────────────────────────────────────────────────────────────────────────
// Sélecteur adaptatif
//
// Justification (Goto & van de Geijn, 2008 ; Drepper, 2007) :
//   - Pour B < THRESHOLD_B  : le coût de fork/join OpenMP (~50–200µs)
//     dépasse le gain de calcul → séquentiel plus rapide
//   - Pour Din*Dout < THRESHOLD_DIM² : la matrice tient entièrement
//     en cache L1/L2 → pas de bénéfice à partitionner
// ─────────────────────────────────────────────────────────────────────────────
bool DenseLayerDataParallel::should_parallelize(int B, int Din, int Dout) const {
    return (B    >= THRESHOLD_B)   &&
           (Din  >= THRESHOLD_DIM  ||
            Dout >= THRESHOLD_DIM);
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward
// ─────────────────────────────────────────────────────────────────────────────
Tensor DenseLayerDataParallel::forward(const Tensor& input) {
    cached_rank = input.ndim();
    input_cache = input;

    Eigen::MatrixXf X = input.toMatrix();   // (B, Din)
    const int B = static_cast<int>(X.rows());

    if (X.cols() != input_size)
        throw std::runtime_error(
            "[DenseLayerDataParallel] forward: input_size mismatch — attendu "
            + std::to_string(input_size) + ", reçu "
            + std::to_string(X.cols()));

    if (should_parallelize(B, input_size, output_size))
        return forward_parallel(X, B);
    else
        return forward_sequential(X, B);
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward séquentiel — GEMM Eigen pur, vectorisé AVX2 via -O3 -mavx2 -mfma
// ─────────────────────────────────────────────────────────────────────────────
Tensor DenseLayerDataParallel::forward_sequential(const Eigen::MatrixXf& X,
                                                   int B) {
    // (B, Din) × (Din, Dout) = (B, Dout)
    Eigen::MatrixXf Y = X * weights.transpose();
    Y.rowwise() += bias.transpose();
    return buildOutput(Y, B);
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward parallèle — tiling manuel pour localité cache L1
//
// Stratégie (Goto & van de Geijn, 2008) :
//   On partitionne B en blocs de TILE_B et Dout en blocs de TILE_D.
//   Chaque thread travaille sur un bloc (ii, jj) indépendant →
//   pas de false sharing car les blocs de Y sont disjoints par thread.
//
//   TILE_B * TILE_D * sizeof(float) = 64 * 128 * 4 = 32 KB ≤ L1 cache
//
// Correction false sharing vs version précédente :
//   collapse(2) avec écriture sur Y(i,j) causait des conflits de ligne
//   de cache entre threads adjacents (ligne cache = 16 floats = 64 bytes).
//   Ici on parallélise uniquement sur ii (lignes de batch) — chaque thread
//   écrit dans des lignes de Y disjointes → 0 false sharing.
// ─────────────────────────────────────────────────────────────────────────────
Tensor DenseLayerDataParallel::forward_parallel(const Eigen::MatrixXf& X,
                                                 int B) {
    const int Dout = output_size;
    Eigen::MatrixXf Y(B, Dout);

#pragma omp parallel for num_threads(n_threads_) schedule(static)
    for (int ii = 0; ii < B; ii += TILE_B) {
        const int i_max = std::min(ii + TILE_B, B);

        for (int jj = 0; jj < Dout; jj += TILE_D) {
            const int j_max = std::min(jj + TILE_D, Dout);

            // Bloc (B_tile × D_tile) — tient en L1
            // noalias() : Eigen ne crée pas de temporaire
            Y.block(ii, jj, i_max - ii, j_max - jj).noalias() =
                X.block(ii, 0, i_max - ii, input_size) *
                weights.block(jj, 0, j_max - jj, input_size).transpose();
        }
    }

    // Biais : chaque thread écrit sa propre ligne → pas de conflit
#pragma omp parallel for num_threads(n_threads_) schedule(static)
    for (int i = 0; i < B; ++i)
        Y.row(i) += bias.transpose();

    return buildOutput(Y, B);
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward
// ─────────────────────────────────────────────────────────────────────────────
Tensor DenseLayerDataParallel::backward(const Tensor& grad_output) {
    Eigen::MatrixXf dO = grad_output.toMatrix();  // (B, Dout)
    Eigen::MatrixXf X  = input_cache.toMatrix();  // (B, Din)
    const int B = static_cast<int>(dO.rows());

    if (should_parallelize(B, input_size, output_size))
        return backward_parallel(dO, X, B);
    else
        return backward_sequential(dO, X, B);
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward séquentiel
// ─────────────────────────────────────────────────────────────────────────────
Tensor DenseLayerDataParallel::backward_sequential(const Eigen::MatrixXf& dO,
                                                    const Eigen::MatrixXf& X,
                                                    int B) {
    grad_weights = (X.transpose() * dO).transpose() / static_cast<float>(B);
    grad_bias    = dO.colwise().sum().transpose()    / static_cast<float>(B);

    Eigen::MatrixXf dX = dO * weights;
    return buildGradInput(dX);
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward parallèle
//
// Stratégie de réduction (Harris, 2007 — tree reduction) :
//   Chaque thread accumule son propre grad_weights local →
//   0 synchronisation pendant le calcul (nowait).
//   Réduction finale séquentielle hors parallel region →
//   pas de race condition, pas de faux partage.
//
//   Complexité réduction : O(n_threads * Dout * Din) additions
//   vs O(B * Dout * Din) pour une réduction naïve → gain si n_threads << B
// ─────────────────────────────────────────────────────────────────────────────
Tensor DenseLayerDataParallel::backward_parallel(const Eigen::MatrixXf& dO,
                                                  const Eigen::MatrixXf& X,
                                                  int B) {
    const float inv_B = 1.0f / static_cast<float>(B);

    // ── 1. grad_weights : réduction par thread ──────────────────────────────
    // Chaque thread a sa propre matrice locale → 0 écriture partagée
    std::vector<Eigen::MatrixXf> local_gw(
        n_threads_, Eigen::MatrixXf::Zero(output_size, input_size));

#pragma omp parallel num_threads(n_threads_)
    {
        const int tid = omp_get_thread_num();
        Eigen::MatrixXf& lgw = local_gw[tid];

        // nowait : pas de barrière implicite en fin de for
        // → les threads passent directement à la suite
#pragma omp for schedule(static) nowait
        for (int i = 0; i < B; ++i)
            lgw.noalias() += dO.row(i).transpose() * X.row(i);
    }
    // Réduction séquentielle finale — hors parallel region → thread-safe
    grad_weights = Eigen::MatrixXf::Zero(output_size, input_size);
    for (const auto& lgw : local_gw)
        grad_weights += lgw;
    grad_weights *= inv_B;

    // ── 2. grad_bias ─────────────────────────────────────────────────────────
    // Réduction sur B : chaque thread somme sa partition de lignes
    std::vector<Eigen::VectorXf> local_gb(
        n_threads_, Eigen::VectorXf::Zero(output_size));

#pragma omp parallel num_threads(n_threads_)
    {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(static) nowait
        for (int i = 0; i < B; ++i)
            local_gb[tid] += dO.row(i).transpose();
    }
    grad_bias = Eigen::VectorXf::Zero(output_size);
    for (const auto& lgb : local_gb)
        grad_bias += lgb;
    grad_bias *= inv_B;

    // ── 3. grad_input : parallélisation sur B ────────────────────────────────
    // Chaque thread écrit dans ses propres lignes de dX → 0 false sharing
    Eigen::MatrixXf dX(B, input_size);
#pragma omp parallel for num_threads(n_threads_) schedule(static)
    for (int b = 0; b < B; ++b)
        dX.row(b).noalias() = dO.row(b) * weights;

    return buildGradInput(dX);
}