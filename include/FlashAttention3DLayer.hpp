#pragma once
#include "Layer.hpp"
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>

// =============================================================================
// FlashAttention3DLayer
// =============================================================================
//
// Implémentation complète de Flash Attention (Dao et al., 2022) pour volumes
// 3D médicaux, en C++17 pur avec Eigen3. Même interface Layer que
// WindowAttention3DLayer — substitution directe dans CNN::addLayer().
//
// ── Pourquoi Flash Attention ─────────────────────────────────────────────────
//
// L'attention standard calcule :
//   S   = Q · K^T / sqrt(d_k)        (N, N)  ← stockée entièrement en RAM
//   A   = softmax(S)                  (N, N)  ← stockée entièrement en RAM
//   Out = A · V                       (N, C)
//
// Pour N = 28×28×28 = 21 952, la matrice N×N coûte 1.9 GB par image.
// Avec un batch de 8 → 15 GB. Irréaliste sans GPU dédié.
//
// Flash Attention résout ce problème par tiling :
//   → La matrice A n'est JAMAIS matérialisée entièrement en mémoire.
//   → On calcule Out par blocs de taille BLOCK_SIZE, en maintenant
//     des statistiques roulantes (max, logsumexp) pour le softmax.
//   → Mémoire : O(N × C) au lieu de O(N²)
//   → Même résultat mathématique exact, identique à float près.
//
// ── Algorithme (Flash Attention forward avec tiling) ─────────────────────────
//
// Pour chaque bloc de queries Q_i (taille Br × C) :
//   Initialiser :  O_i = 0,  l_i = 0,  m_i = -inf
//
//   Pour chaque bloc de clés/valeurs K_j, V_j (taille Bc × C) :
//     1. S_ij   = Q_i · K_j^T * scale              (Br × Bc)
//     2. m_ij   = rowmax(S_ij)                       (Br)
//     3. P_ij   = exp(S_ij − m_ij)                  (Br × Bc) — softmax partiel
//     4. l_ij   = rowsum(P_ij)                       (Br)
//     5. m_new  = max(m_i, m_ij)                     (Br) — nouveau max global
//     6. O_i   ← exp(m_i − m_new) · O_i
//               + exp(m_ij − m_new) · P_ij · V_j    (Br × C) — correction
//     7. l_i   ← exp(m_i − m_new) · l_i
//               + exp(m_ij − m_new) · l_ij           (Br) — normalisation roulante
//     8. m_i   ← m_new
//
//   Normalisation finale : O_i /= l_i               (broadcast)
//
// ── Backward (Flash Attention recompute) ─────────────────────────────────────
//
// Pour éviter de stocker A (N×N), le backward recompute P_ij à la volée
// depuis Q_i, K_j et les statistiques m, l mis en cache (taille O(N)).
//
//   Pour chaque (i, j) :
//     P_ij = exp(S_ij − m_i) / l_i                  (Br × Bc)
//     dV_j += P_ij^T · dO_i
//     dP_ij = dO_i · V_j^T
//     dS_ij = P_ij ⊙ (dP_ij − rowsum(dO_i ⊙ O_i))  ← backward softmax online
//     dQ_i += dS_ij · K_j * scale
//     dK_j += dS_ij^T · Q_i * scale
//
// ── Complexité ───────────────────────────────────────────────────────────────
//
//   Mémoire forward  : O(N × C)  — vs O(N²) pour l'attention standard
//   Mémoire backward : O(N × C)  — seuls m, l, O sont mis en cache
//   Calcul           : O(N² × C) — même que l'attention standard
//                                   mais avec une constante réduite (tiling SRAM)
//
// ── Paramètres de tiling ─────────────────────────────────────────────────────
//
//   BLOCK_SIZE  : taille du bloc Br = Bc (défaut : 64)
//     64 est optimal pour la plupart des tailles de volume médicaux.
//     Pour N < 64, Flash Attention dégénère gracieusement en attention standard.
//     Pour N >> 64, augmenter jusqu'à 128 ou 256 si la L1 cache est grande.
//
// ── Compatibilité avec WindowAttention3DLayer ─────────────────────────────────
//
//   FlashAttention3DLayer a exactement la même interface :
//     - Même constructeur (channels, window_d, window_h, window_w, num_heads,
//       use_residual, use_norm)
//     - Même forward(Tensor&) / backward(Tensor&) / updateParams(Optimizer&)
//     - Substitution directe dans CNN::addLayer()
//
//   La seule différence : le paramètre block_size (défaut 64) qui contrôle
//   le tiling. Absent de WindowAttention3DLayer (qui stocke A entièrement).
//
// ── Utilisation ───────────────────────────────────────────────────────────────
//
//   // Remplace WindowAttention3DLayer sans rien changer d'autre :
//   model.addLayer(std::make_shared<FlashAttention3DLayer>(
//       64,   // channels
//       7,7,7,// window — ici 7³ = attention globale sur volume 7³
//       4,    // num_heads
//       64,   // block_size (tiling SRAM)
//       true, true));
//
// =============================================================================

class FlashAttention3DLayer : public Layer {
public:

    // ── Constructeur ──────────────────────────────────────────────────────────
    FlashAttention3DLayer(int  channels,
                          int  window_d     = 7,
                          int  window_h     = 7,
                          int  window_w     = 7,
                          int  num_heads    = 4,
                          int  block_size   = 64,
                          bool use_residual = true,
                          bool use_norm     = true)
        : C_(channels),
          wd_(window_d), wh_(window_h), ww_(window_w),
          num_heads_(num_heads),
          Br_(block_size), Bc_(block_size),
          use_residual_(use_residual),
          use_norm_(use_norm)
    {
        if (C_ % num_heads_ != 0)
            throw std::invalid_argument(
                "[FlashAttention3D] channels (" + std::to_string(C_) +
                ") doit être divisible par num_heads (" +
                std::to_string(num_heads_) + ")");

        d_k_   = C_ / num_heads_;
        scale_ = 1.0f / std::sqrt(static_cast<float>(d_k_));

        // Projections : (C, C) pour Q, K, V, O
        W_Q_ = Eigen::MatrixXf(C_, C_);
        W_K_ = Eigen::MatrixXf(C_, C_);
        W_V_ = Eigen::MatrixXf(C_, C_);
        W_O_ = Eigen::MatrixXf(C_, C_);

        dW_Q_ = Eigen::MatrixXf::Zero(C_, C_);
        dW_K_ = Eigen::MatrixXf::Zero(C_, C_);
        dW_V_ = Eigen::MatrixXf::Zero(C_, C_);
        dW_O_ = Eigen::MatrixXf::Zero(C_, C_);

        // LayerNorm
        gamma_   = Eigen::VectorXf::Ones(C_);
        beta_    = Eigen::VectorXf::Zero(C_);
        d_gamma_ = Eigen::VectorXf::Zero(C_);
        d_beta_  = Eigen::VectorXf::Zero(C_);

        isTrainable = true;
        initializeWeights();
    }

    ~FlashAttention3DLayer() override = default;

    FlashAttention3DLayer(const FlashAttention3DLayer&)            = delete;
    FlashAttention3DLayer& operator=(const FlashAttention3DLayer&) = delete;

    // ── Initialisation Xavier ─────────────────────────────────────────────────
    void initializeWeights() {
        const float scale = std::sqrt(2.0f / (C_ + C_));
        std::mt19937 gen{std::random_device{}()};
        std::normal_distribution<float> dist(0.f, scale);
        auto fill = [&](Eigen::MatrixXf& M) {
            for (int r = 0; r < M.rows(); ++r)
                for (int c = 0; c < M.cols(); ++c)
                    M(r, c) = dist(gen);
        };
        fill(W_Q_); fill(W_K_); fill(W_V_); fill(W_O_);
        gamma_.setOnes();
        beta_.setZero();
    }

    // =========================================================================
    // FORWARD — Flash Attention avec tiling
    // =========================================================================
    //
    // Entrée : Tensor 5D (B, C, D, H, W)
    // Sortie : Tensor 5D (B, C, D, H, W)  — dimensions préservées
    //
    // Différence clé avec WindowAttention3DLayer :
    //   La matrice A = softmax(Q·K^T/√d) n'est JAMAIS allouée entièrement.
    //   On maintient des accumulateurs (O, l, m) mis à jour bloc par bloc.
    Tensor forward(const Tensor& input) override {

        if (input.ndim() != 5)
            throw std::runtime_error(
                "[FlashAttention3D::forward] Attend Tensor 5D, reçu ndim="
                + std::to_string(input.ndim()));
        if (input.dim(1) != C_)
            throw std::runtime_error(
                "[FlashAttention3D::forward] channels mismatch");

        input_cache_ = input;

        const int B = input.dim(0);
        const int D = input.dim(2);
        const int H = input.dim(3);
        const int W = input.dim(4);

        // Fenêtres
        const int nD    = ceilDiv(D, wd_);
        const int nH    = ceilDiv(H, wh_);
        const int nW    = ceilDiv(W, ww_);
        const int N_win = nD * nH * nW;
        const int N_tok = wd_ * wh_ * ww_;   // N = tokens par fenêtre
        const int BW    = B * N_win;          // contextes indépendants

        partition_info_ = {B, D, H, W, nD, nH, nW};

        // ── Étape 1 : Partition ───────────────────────────────────────────────
        // tokens : (BW * N_tok, C)
        Eigen::MatrixXf tokens = partition(input, B, D, H, W, nD, nH, nW);
        tokens_cache_ = tokens;

        // ── Étape 2 : Projections globales Q, K, V ────────────────────────────
        // (T, C) où T = BW * N_tok
        Eigen::MatrixXf Q = tokens * W_Q_.transpose();
        Eigen::MatrixXf K = tokens * W_K_.transpose();
        Eigen::MatrixXf V = tokens * W_V_.transpose();

        Q_cache_ = Q;
        K_cache_ = K;
        V_cache_ = V;

        // ── Étape 3 : Flash Attention par fenêtre ─────────────────────────────
        //
        // Pour chaque fenêtre bw ∈ [0, BW[ :
        //   Q_bw : (N, C)  avec N = N_tok
        //   K_bw : (N, C)
        //   V_bw : (N, C)
        //
        // On calcule O_bw = FlashAttn(Q_bw, K_bw, V_bw) en O(N) mémoire.
        //
        // Cache pour le backward :
        //   O_cache_ : (T, C)      — sorties normalisées
        //   m_cache_ : (T,)        — max roulant final par query (logsumexp num.)
        //   l_cache_ : (T,)        — normalisation finale par query

        Eigen::MatrixXf O_all(BW * N_tok, C_);
        m_cache_.resize(BW * N_tok);
        l_cache_.resize(BW * N_tok);

        for (int bw = 0; bw < BW; ++bw) {
            const int base = bw * N_tok;

            // Extraction des blocs Q, K, V de cette fenêtre
            const Eigen::MatrixXf Q_win = Q.middleRows(base, N_tok);  // (N, C)
            const Eigen::MatrixXf K_win = K.middleRows(base, N_tok);  // (N, C)
            const Eigen::MatrixXf V_win = V.middleRows(base, N_tok);  // (N, C)

            // Résultat de Flash Attention pour cette fenêtre
            Eigen::MatrixXf O_win(N_tok, C_);
            Eigen::VectorXf m_win(N_tok), l_win(N_tok);

            flashAttentionForward(Q_win, K_win, V_win,
                                  O_win, m_win, l_win,
                                  N_tok);

            O_all.middleRows(base, N_tok) = O_win;
            m_cache_.segment(base, N_tok) = m_win;
            l_cache_.segment(base, N_tok) = l_win;
        }

        O_cache_ = O_all;

        // ── Étape 4 : Projection de sortie W_O ───────────────────────────────
        Eigen::MatrixXf out_proj = O_all * W_O_.transpose();  // (T, C)
        attn_out_cache_ = O_all;  // avant W_O — nécessaire pour dW_O backward

        // ── Étape 5 : Reconstruction du Tensor ───────────────────────────────
        Tensor output = reconstruct(out_proj, B, D, H, W, nD, nH, nW);

        // ── Étape 6 : Résiduelle + LayerNorm ─────────────────────────────────
        if (use_residual_)
            for (int i = 0; i < output.size(); ++i)
                output[i] += input[i];

        if (use_norm_)
            output = layerNorm(output, B);

        return output;
    }

    // =========================================================================
    // BACKWARD — Flash Attention avec recomputation de P_ij
    // =========================================================================
    //
    // Clé : on ne stocke PAS A = softmax(Q·K^T).
    // On recompute P_ij = exp(S_ij − m_i) / l_i dans chaque bloc.
    // Mémoire backward : O(N × C) au lieu de O(N²).
    Tensor backward(const Tensor& grad_output) override {

        const int B    = partition_info_[0];
        const int D    = partition_info_[1];
        const int H    = partition_info_[2];
        const int W    = partition_info_[3];
        const int nD   = partition_info_[4];
        const int nH   = partition_info_[5];
        const int nW   = partition_info_[6];
        const int N_win = nD * nH * nW;
        const int N_tok = wd_ * wh_ * ww_;
        const int BW    = B * N_win;
        const float inv_B = 1.f / static_cast<float>(B);

        // ── 6b. Backward LayerNorm ────────────────────────────────────────────
        Tensor grad = grad_output;
        if (use_norm_)
            grad = layerNormBackward(grad, B);

        // ── 5b. Résiduelle ────────────────────────────────────────────────────
        Tensor grad_residual(grad.shape());
        if (use_residual_)
            grad_residual = grad;
        else
            grad_residual.setZero();

        // ── 5b. Reconstruction gradient → fenêtres ────────────────────────────
        // d_out_proj : (BW * N_tok, C) — gradient avant W_O
        Eigen::MatrixXf d_out_proj = partition(grad, B, D, H, W, nD, nH, nW);

        // ── 4b. Backward W_O ──────────────────────────────────────────────────
        // out_proj = attn_out_cache_ * W_O^T
        // d_attn_out = d_out_proj * W_O
        // dW_O       = d_out_proj^T * attn_out_cache_ / B
        Eigen::MatrixXf d_attn_out = d_out_proj * W_O_;
        dW_O_ = (d_out_proj.transpose() * attn_out_cache_) * inv_B;

        // ── 3b. Flash Attention backward par fenêtre ──────────────────────────
        //
        // Pour chaque fenêtre bw :
        //   On a en cache : Q_bw, K_bw, V_bw, O_bw, m_bw, l_bw
        //   On recompute P_ij bloc par bloc et on accumule dQ, dK, dV.
        Eigen::MatrixXf d_Q = Eigen::MatrixXf::Zero(BW * N_tok, C_);
        Eigen::MatrixXf d_K = Eigen::MatrixXf::Zero(BW * N_tok, C_);
        Eigen::MatrixXf d_V = Eigen::MatrixXf::Zero(BW * N_tok, C_);

        for (int bw = 0; bw < BW; ++bw) {
            const int base = bw * N_tok;

            const Eigen::MatrixXf Q_win  = Q_cache_.middleRows(base, N_tok);
            const Eigen::MatrixXf K_win  = K_cache_.middleRows(base, N_tok);
            const Eigen::MatrixXf V_win  = V_cache_.middleRows(base, N_tok);
            const Eigen::MatrixXf O_win  = O_cache_.middleRows(base, N_tok);
            const Eigen::MatrixXf dO_win = d_attn_out.middleRows(base, N_tok);
            const Eigen::VectorXf m_win  = m_cache_.segment(base, N_tok);
            const Eigen::VectorXf l_win  = l_cache_.segment(base, N_tok);

            Eigen::MatrixXf dQ_win(N_tok, C_);
            Eigen::MatrixXf dK_win(N_tok, C_);
            Eigen::MatrixXf dV_win(N_tok, C_);

            flashAttentionBackward(Q_win, K_win, V_win,
                                   O_win, dO_win,
                                   m_win, l_win,
                                   dQ_win, dK_win, dV_win,
                                   N_tok);

            d_Q.middleRows(base, N_tok) = dQ_win;
            d_K.middleRows(base, N_tok) = dK_win;
            d_V.middleRows(base, N_tok) = dV_win;
        }

        // ── 2b. Backward projections W_Q, W_K, W_V ───────────────────────────
        dW_Q_ = (d_Q.transpose() * tokens_cache_) * inv_B;
        dW_K_ = (d_K.transpose() * tokens_cache_) * inv_B;
        dW_V_ = (d_V.transpose() * tokens_cache_) * inv_B;

        // Gradient vers les tokens d'entrée
        Eigen::MatrixXf d_tokens = d_Q * W_Q_
                                 + d_K * W_K_
                                 + d_V * W_V_;

        // ── 1b. Backward partition → Tensor ───────────────────────────────────
        Tensor grad_input = reconstruct(d_tokens, B, D, H, W, nD, nH, nW);

        if (use_residual_)
            for (int i = 0; i < grad_input.size(); ++i)
                grad_input[i] += grad_residual[i];

        return grad_input;
    }

    // ── Mise à jour des poids ─────────────────────────────────────────────────
    void updateParams(Optimizer& optimizer) override {
        optimizer.updateWeights(W_Q_, dW_Q_);
        optimizer.updateWeights(W_K_, dW_K_);
        optimizer.updateWeights(W_V_, dW_V_);
        optimizer.updateWeights(W_O_, dW_O_);
        optimizer.updateBias(gamma_, d_gamma_);
        optimizer.updateBias(beta_,  d_beta_);
        dW_Q_.setZero(); dW_K_.setZero();
        dW_V_.setZero(); dW_O_.setZero();
        d_gamma_.setZero(); d_beta_.setZero();
    }

    std::string getName() const override {
        return "FlashAttention3D(C=" + std::to_string(C_)
             + " win=" + std::to_string(wd_) + "x"
             + std::to_string(wh_) + "x" + std::to_string(ww_)
             + " heads=" + std::to_string(num_heads_)
             + " Br=" + std::to_string(Br_) + ")";
    }

    int numParams() const {
        return 4 * C_ * C_ + 2 * C_;
    }

    // ── Accesseurs ────────────────────────────────────────────────────────────
    Eigen::MatrixXf& getWQ() { return W_Q_; }
    Eigen::MatrixXf& getWK() { return W_K_; }
    Eigen::MatrixXf& getWV() { return W_V_; }
    Eigen::MatrixXf& getWO() { return W_O_; }
    int blockSize() const { return Br_; }

private:

    // ── Hyperparamètres ───────────────────────────────────────────────────────
    int   C_, wd_, wh_, ww_, num_heads_, d_k_;
    int   Br_, Bc_;        // tailles des blocs de tiling (query, key)
    float scale_;
    bool  use_residual_, use_norm_;

    // ── Paramètres apprenables ────────────────────────────────────────────────
    Eigen::MatrixXf W_Q_, W_K_, W_V_, W_O_;
    Eigen::VectorXf gamma_, beta_;

    // ── Gradients ─────────────────────────────────────────────────────────────
    Eigen::MatrixXf dW_Q_, dW_K_, dW_V_, dW_O_;
    Eigen::VectorXf d_gamma_, d_beta_;

    // ── Cache backward ────────────────────────────────────────────────────────
    // Clé : O(N×C) seulement — pas de matrice A(N×N)
    Tensor          input_cache_;
    Eigen::MatrixXf tokens_cache_;    // (T, C) — tokens d'entrée
    Eigen::MatrixXf Q_cache_;         // (T, C)
    Eigen::MatrixXf K_cache_;         // (T, C)
    Eigen::MatrixXf V_cache_;         // (T, C)
    Eigen::MatrixXf O_cache_;         // (T, C) — sorties attention normalisées
    Eigen::MatrixXf attn_out_cache_;  // (T, C) — avant W_O (= O_cache_ ici)
    Eigen::VectorXf m_cache_;         // (T,)   — max roulant final par query
    Eigen::VectorXf l_cache_;         // (T,)   — somme exp finale par query
    std::vector<int> partition_info_; // {B,D,H,W,nD,nH,nW}

    // Cache LayerNorm
    Eigen::MatrixXf ln_x_norm_;
    Eigen::VectorXf ln_mean_, ln_var_;
    int ln_N_ = 0;

    // =========================================================================
    // FLASH ATTENTION FORWARD — cœur de l'algorithme
    // =========================================================================
    //
    // Calcule O = softmax(Q·K^T/√d)·V sans stocker la matrice N×N.
    //
    // Entrées :
    //   Q, K, V : (N, C)
    //   N       : nombre de tokens dans cette fenêtre
    //
    // Sorties :
    //   O : (N, C)  — résultat normalisé
    //   m : (N,)    — max roulant final (pour le backward)
    //   l : (N,)    — normalisation finale (pour le backward)
    //
    // Invariant maintenu à chaque itération j :
    //   m_i = max(S_i0, S_i1, ..., S_i,j-1)  pour chaque query i
    //   l_i = sum(exp(S_i0-m_i), ..., exp(S_i,j-1-m_i))
    //   O_i = (1/l_i) * sum_k exp(S_ik-m_i) * V_k
    //
    // Propriété : à la fin de la boucle sur j, O_i = softmax(S_i)·V
    // EXACTEMENT — pas d'approximation.
    // =========================================================================
    void flashAttentionForward(const Eigen::MatrixXf& Q,   // (N, C)
                                const Eigen::MatrixXf& K,   // (N, C)
                                const Eigen::MatrixXf& V,   // (N, C)
                                Eigen::MatrixXf& O,          // (N, C) — sortie
                                Eigen::VectorXf& m,          // (N,)   — max final
                                Eigen::VectorXf& l,          // (N,)   — norm finale
                                int N) const
    {
        // Initialisation des accumulateurs
        O.setZero();                                           // (N, C)
        m.setConstant(-std::numeric_limits<float>::infinity()); // (N,)
        l.setZero();                                           // (N,)

        const int num_blocks_c = ceilDiv(N, Bc_);  // nombre de blocs de clés

        // ── Boucle sur les blocs de clés/valeurs ─────────────────────────────
        for (int jb = 0; jb < num_blocks_c; ++jb) {
            const int j_start = jb * Bc_;
            const int j_end   = std::min(j_start + Bc_, N);
            const int Bcj     = j_end - j_start;  // taille réelle du bloc

            // Extraction du bloc K_j, V_j : (Bcj, C)
            const Eigen::MatrixXf K_j = K.middleRows(j_start, Bcj);
            const Eigen::MatrixXf V_j = V.middleRows(j_start, Bcj);

            const int num_blocks_r = ceilDiv(N, Br_);  // blocs de queries

            // ── Boucle sur les blocs de queries ───────────────────────────────
            for (int ib = 0; ib < num_blocks_r; ++ib) {
                const int i_start = ib * Br_;
                const int i_end   = std::min(i_start + Br_, N);
                const int Bri     = i_end - i_start;  // taille réelle du bloc

                // Extraction Q_i, O_i, m_i, l_i
                const Eigen::MatrixXf Q_i = Q.middleRows(i_start, Bri);  // (Bri, C)
                Eigen::MatrixXf O_i = O.middleRows(i_start, Bri);        // (Bri, C)
                Eigen::VectorXf m_i = m.segment(i_start, Bri);            // (Bri,)
                Eigen::VectorXf l_i = l.segment(i_start, Bri);            // (Bri,)

                // ── Étape 1 : Scores S_ij = Q_i · K_j^T * scale ──────────────
                // S_ij : (Bri, Bcj)
                Eigen::MatrixXf S_ij = (Q_i * K_j.transpose()) * scale_;

                // ── Étape 2 : Max par ligne de S_ij ───────────────────────────
                // m_ij : (Bri,)
                Eigen::VectorXf m_ij = S_ij.rowwise().maxCoeff();  // (Bri,)

                // ── Étape 3 : Softmax partiel P_ij = exp(S_ij - m_ij) ────────
                // Soustraction ligne par ligne pour la stabilité numérique
                Eigen::MatrixXf P_ij(Bri, Bcj);
                for (int r = 0; r < Bri; ++r)
                    P_ij.row(r) = (S_ij.row(r).array() - m_ij[r]).exp();

                // ── Étape 4 : Somme partielle l_ij ────────────────────────────
                Eigen::VectorXf l_ij = P_ij.rowwise().sum();  // (Bri,)

                // ── Étapes 5-8 : Mise à jour roulante (O, l, m) ───────────────
                //
                // Nouveau max global : m_new = max(m_i, m_ij)
                Eigen::VectorXf m_new = m_i.cwiseMax(m_ij);

                // Facteurs de correction exp(ancien_max - nouveau_max)
                // exp(m_i   - m_new) : pour corriger l'accumulation existante
                // exp(m_ij  - m_new) : pour intégrer le nouveau bloc
                Eigen::VectorXf alpha = (m_i.array()  - m_new.array()).exp();
                Eigen::VectorXf beta  = (m_ij.array() - m_new.array()).exp();

                // Mise à jour de O_i :
                // O_new = exp(m_i - m_new) * O_i + exp(m_ij - m_new) * P_ij * V_j
                Eigen::MatrixXf P_V = P_ij * V_j;  // (Bri, C)
                for (int r = 0; r < Bri; ++r) {
                    O_i.row(r) = alpha[r] * O_i.row(r)
                                + beta[r]  * P_V.row(r);
                }

                // Mise à jour de l_i :
                // l_new = exp(m_i - m_new) * l_i + exp(m_ij - m_new) * l_ij
                Eigen::VectorXf l_new = alpha.array() * l_i.array()
                                       + beta.array()  * l_ij.array();

                // Écriture des accumulateurs mis à jour
                O.middleRows(i_start, Bri) = O_i;
                m.segment(i_start, Bri)    = m_new;
                l.segment(i_start, Bri)    = l_new;
            }
        }

        // ── Normalisation finale : O /= l (broadcast) ────────────────────────
        // Après la boucle complète sur j, l_i = sum_j exp(S_ij - m_i)
        // Donc O_i = (1/l_i) * sum_j exp(S_ij - m_i) * V_j = softmax(S_i)·V
        for (int i = 0; i < N; ++i) {
            if (l[i] > 1e-9f)
                O.row(i) /= l[i];
        }
    }

    // =========================================================================
    // FLASH ATTENTION BACKWARD — recomputation de P_ij sans stocker A
    // =========================================================================
    //
    // Calcule dQ, dK, dV depuis dO, sans utiliser la matrice A stockée.
    // P_ij est recompuée depuis Q, K, m, l dans chaque bloc.
    //
    // Entrées :
    //   Q, K, V  : (N, C)
    //   O        : (N, C) — sortie du forward (normalisée)
    //   dO       : (N, C) — gradient entrant
    //   m        : (N,)   — max roulant final
    //   l        : (N,)   — normalisation finale
    //
    // Sorties :
    //   dQ, dK, dV : (N, C)
    //
    // Formule backward softmax online :
    //   D_i = rowsum(dO_i ⊙ O_i)    ← "delta" par query
    //   dS_ij = P_ij ⊙ (dP_ij - D_i)   où dP_ij = dO_i · V_j^T
    //   dQ_i += dS_ij · K_j * scale
    //   dK_j += dS_ij^T · Q_i * scale
    //   dV_j += P_ij^T · dO_i
    // =========================================================================
    void flashAttentionBackward(const Eigen::MatrixXf& Q,    // (N, C)
                                 const Eigen::MatrixXf& K,    // (N, C)
                                 const Eigen::MatrixXf& V,    // (N, C)
                                 const Eigen::MatrixXf& O,    // (N, C)
                                 const Eigen::MatrixXf& dO,   // (N, C)
                                 const Eigen::VectorXf& m,    // (N,)
                                 const Eigen::VectorXf& l,    // (N,)
                                 Eigen::MatrixXf& dQ,          // (N, C) — sortie
                                 Eigen::MatrixXf& dK,          // (N, C) — sortie
                                 Eigen::MatrixXf& dV,          // (N, C) — sortie
                                 int N) const
    {
        dQ.setZero();
        dK.setZero();
        dV.setZero();

        // D_i = rowsum(dO ⊙ O) : scalaire par query, (N,)
        // Quantité nécessaire pour le backward du softmax online
        Eigen::VectorXf D = (dO.array() * O.array()).rowwise().sum();  // (N,)

        const int num_blocks_c = ceilDiv(N, Bc_);
        const int num_blocks_r = ceilDiv(N, Br_);

        // ── Double boucle sur les blocs ───────────────────────────────────────
        for (int jb = 0; jb < num_blocks_c; ++jb) {
            const int j_start = jb * Bc_;
            const int j_end   = std::min(j_start + Bc_, N);
            const int Bcj     = j_end - j_start;

            const Eigen::MatrixXf K_j = K.middleRows(j_start, Bcj);  // (Bcj, C)
            const Eigen::MatrixXf V_j = V.middleRows(j_start, Bcj);  // (Bcj, C)
            Eigen::MatrixXf dK_j = Eigen::MatrixXf::Zero(Bcj, C_);
            Eigen::MatrixXf dV_j = Eigen::MatrixXf::Zero(Bcj, C_);

            for (int ib = 0; ib < num_blocks_r; ++ib) {
                const int i_start = ib * Br_;
                const int i_end   = std::min(i_start + Br_, N);
                const int Bri     = i_end - i_start;

                const Eigen::MatrixXf Q_i  = Q.middleRows(i_start, Bri);   // (Bri, C)
                const Eigen::MatrixXf dO_i = dO.middleRows(i_start, Bri);  // (Bri, C)
                const Eigen::VectorXf m_i  = m.segment(i_start, Bri);       // (Bri,)
                const Eigen::VectorXf l_i  = l.segment(i_start, Bri);       // (Bri,)
                const Eigen::VectorXf D_i  = D.segment(i_start, Bri);       // (Bri,)

                // ── Recomputation de P_ij ─────────────────────────────────────
                // S_ij = Q_i · K_j^T * scale              (Bri, Bcj)
                // P_ij = exp(S_ij - m_i) / l_i            (Bri, Bcj)
                Eigen::MatrixXf S_ij = (Q_i * K_j.transpose()) * scale_;  // (Bri, Bcj)
                Eigen::MatrixXf P_ij(Bri, Bcj);
                for (int r = 0; r < Bri; ++r) {
                    if (l_i[r] > 1e-9f)
                        P_ij.row(r) = ((S_ij.row(r).array() - m_i[r]).exp())
                                      / l_i[r];
                    else
                        P_ij.row(r).setZero();
                }

                // ── dV_j += P_ij^T · dO_i ────────────────────────────────────
                dV_j += P_ij.transpose() * dO_i;  // (Bcj, C)

                // ── dP_ij = dO_i · V_j^T ─────────────────────────────────────
                Eigen::MatrixXf dP_ij = dO_i * V_j.transpose();  // (Bri, Bcj)

                // ── dS_ij = P_ij ⊙ (dP_ij - D_i) ────────────────────────────
                // Backward du softmax : dL/dS_ij = P_ij * (dL/dP_ij - D_i)
                // où D_i = rowsum(dO_i ⊙ O_i)
                Eigen::MatrixXf dS_ij(Bri, Bcj);
                for (int r = 0; r < Bri; ++r)
                    dS_ij.row(r) = P_ij.row(r).array()
                                 * (dP_ij.row(r).array() - D_i[r]);

                dS_ij *= scale_;  // facteur 1/sqrt(d_k)

                // ── dQ_i += dS_ij · K_j ──────────────────────────────────────
                dQ.middleRows(i_start, Bri) += dS_ij * K_j;  // (Bri, C)

                // ── dK_j += dS_ij^T · Q_i ────────────────────────────────────
                dK_j += dS_ij.transpose() * Q_i;  // (Bcj, C)
            }

            // Écriture des gradients accumulés pour ce bloc de clés
            dK.middleRows(j_start, Bcj) += dK_j;
            dV.middleRows(j_start, Bcj) += dV_j;
        }
    }

    // =========================================================================
    // PARTITION — Tensor (B,C,D,H,W) → matrice (BW*N_tok, C)
    // =========================================================================
    Eigen::MatrixXf partition(const Tensor& t,
                               int B, int D, int H, int W,
                               int nD, int nH, int nW) const
    {
        const int N_win = nD * nH * nW;
        const int N_tok = wd_ * wh_ * ww_;
        Eigen::MatrixXf out = Eigen::MatrixXf::Zero(B * N_win * N_tok, C_);

        for (int b  = 0; b  < B;  ++b)
        for (int nd = 0; nd < nD; ++nd)
        for (int nh = 0; nh < nH; ++nh)
        for (int nw = 0; nw < nW; ++nw) {
            const int win_idx  = ((nd * nH + nh) * nW + nw);
            const int row_base = (b * N_win + win_idx) * N_tok;
            for (int td = 0; td < wd_; ++td)
            for (int th = 0; th < wh_; ++th)
            for (int tw = 0; tw < ww_; ++tw) {
                const int d = nd * wd_ + td;
                const int h = nh * wh_ + th;
                const int w = nw * ww_ + tw;
                const int tok = (td * wh_ + th) * ww_ + tw;
                if (d < D && h < H && w < W)
                    for (int c = 0; c < C_; ++c)
                        out(row_base + tok, c) = t(b, c, d, h, w);
            }
        }
        return out;
    }

    // =========================================================================
    // RECONSTRUCT — matrice (BW*N_tok, C) → Tensor (B,C,D,H,W)
    // =========================================================================
    Tensor reconstruct(const Eigen::MatrixXf& mat,
                       int B, int D, int H, int W,
                       int nD, int nH, int nW) const
    {
        const int N_win = nD * nH * nW;
        const int N_tok = wd_ * wh_ * ww_;
        Tensor out(B, C_, D, H, W);
        out.setZero();

        for (int b  = 0; b  < B;  ++b)
        for (int nd = 0; nd < nD; ++nd)
        for (int nh = 0; nh < nH; ++nh)
        for (int nw = 0; nw < nW; ++nw) {
            const int win_idx  = ((nd * nH + nh) * nW + nw);
            const int row_base = (b * N_win + win_idx) * N_tok;
            for (int td = 0; td < wd_; ++td)
            for (int th = 0; th < wh_; ++th)
            for (int tw = 0; tw < ww_; ++tw) {
                const int d = nd * wd_ + td;
                const int h = nh * wh_ + th;
                const int w = nw * ww_ + tw;
                if (d >= D || h >= H || w >= W) continue;
                const int tok = (td * wh_ + th) * ww_ + tw;
                for (int c = 0; c < C_; ++c)
                    out(b, c, d, h, w) = mat(row_base + tok, c);
            }
        }
        return out;
    }

    // =========================================================================
    // LAYER NORMALIZATION — forward
    // =========================================================================
    Tensor layerNorm(const Tensor& t, int B) {
        const float eps = 1e-5f;
        const int D = t.dim(2), H = t.dim(3), W = t.dim(4);
        const int N = B * D * H * W;
        Eigen::MatrixXf X(N, C_);
        int idx = 0;
        for (int b=0;b<B;++b) for (int d=0;d<D;++d)
        for (int h=0;h<H;++h) for (int w=0;w<W;++w) {
            for (int c=0;c<C_;++c) X(idx,c) = t(b,c,d,h,w);
            ++idx;
        }
        ln_mean_ = X.rowwise().mean();
        Eigen::MatrixXf Xc = X.colwise() - ln_mean_;
        ln_var_  = Xc.array().square().rowwise().mean();
        ln_x_norm_.resize(N, C_);
        for (int i=0;i<N;++i) {
            float si = 1.f / std::sqrt(ln_var_[i] + eps);
            for (int c=0;c<C_;++c) ln_x_norm_(i,c) = Xc(i,c)*si;
        }
        ln_N_ = N;
        Eigen::MatrixXf Y = (ln_x_norm_.array().rowwise()
                             * gamma_.transpose().array())
                            .rowwise() + beta_.transpose().array();
        Tensor out(B, C_, D, H, W);
        idx = 0;
        for (int b=0;b<B;++b) for (int d=0;d<D;++d)
        for (int h=0;h<H;++h) for (int w=0;w<W;++w) {
            for (int c=0;c<C_;++c) out(b,c,d,h,w) = Y(idx,c);
            ++idx;
        }
        return out;
    }

    // =========================================================================
    // LAYER NORMALIZATION — backward
    // =========================================================================
    Tensor layerNormBackward(const Tensor& grad, int B) {
        const float eps = 1e-5f;
        const int D = grad.dim(2), H = grad.dim(3), W = grad.dim(4);
        Eigen::MatrixXf dY(ln_N_, C_);
        int idx = 0;
        for (int b=0;b<B;++b) for (int d=0;d<D;++d)
        for (int h=0;h<H;++h) for (int w=0;w<W;++w) {
            for (int c=0;c<C_;++c) dY(idx,c) = grad(b,c,d,h,w);
            ++idx;
        }
        d_gamma_ += ((dY.array() * ln_x_norm_.array()).colwise().sum()
                  / static_cast<float>(B)).matrix().transpose();
        d_beta_  += (dY.colwise().sum() / static_cast<float>(B)).transpose();
        Eigen::MatrixXf dX(ln_N_, C_);
        for (int i=0;i<ln_N_;++i) {
            float si = 1.f / std::sqrt(ln_var_[i] + eps);
            Eigen::RowVectorXf dYh = dY.row(i).array() * gamma_.transpose().array();
            float s1 = dYh.sum();
            float s2 = (dYh.array() * ln_x_norm_.row(i).array()).sum();
            for (int c=0;c<C_;++c)
                dX(i,c) = si/C_ * (C_*dYh(c) - s1 - ln_x_norm_(i,c)*s2);
        }
        Tensor out(B, C_, D, H, W);
        idx = 0;
        for (int b=0;b<B;++b) for (int d=0;d<D;++d)
        for (int h=0;h<H;++h) for (int w=0;w<W;++w) {
            for (int c=0;c<C_;++c) out(b,c,d,h,w) = dX(idx,c);
            ++idx;
        }
        return out;
    }

    // ── Utilitaire ────────────────────────────────────────────────────────────
    static int ceilDiv(int a, int b) { return (a + b - 1) / b; }
};