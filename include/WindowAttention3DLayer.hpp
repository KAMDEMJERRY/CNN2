#pragma once
#include "Layer.hpp"
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>

// =============================================================================
// WindowAttention3DLayer — Attention Spatio-Temporelle sur Fenêtres Locales
// =============================================================================
//
// Principe (inspiré de Swin Transformer 3D / Video Swin Transformer) :
//
//   Au lieu de calculer l'attention sur tout le volume (coût O(N²) avec
//   N = D×H×W = 21952 pour 28³), le volume est découpé en fenêtres locales
//   non-chevauchantes de taille (wd, wh, ww). L'attention est calculée
//   indépendamment dans chaque fenêtre.
//
//   Pour un volume (B, C, 7, 7, 7) avec fenêtres 4×4×4 :
//     N_tokens_par_fenetre = 4×4×4 = 64
//     N_fenetres = ceil(7/4) × ceil(7/4) × ceil(7/4) = 2×2×2 = 8
//     Matrice d'attention : 64×64 par fenêtre (vs 343×343 global)
//     Coût : B × 8 × 64² = 32768 × B  (vs B × 343² = 117 649 × B)
//
// Algorithme (un bloc d'attention) :
//
//   1. Partition  : (B, C, D, H, W) → (N_win, N_tok, C)
//      Chaque fenêtre devient un "batch" de tokens.
//
//   2. Projections Q, K, V
//      Q = X · W_Q    (N_win · B, N_tok, d_k)
//      K = X · W_K    (N_win · B, N_tok, d_k)
//      V = X · W_V    (N_win · B, N_tok, d_v)
//
//   3. Attention scores
//      S = (Q · K^T) / sqrt(d_k)    (N_win · B, N_tok, N_tok)
//      A = softmax(S, axis=-1)       stabilisé numériquement
//
//   4. Agrégation
//      Out = A · V                   (N_win · B, N_tok, d_v)
//      Out = Out · W_O               projection finale → C canaux
//
//   5. Reconstruction : (N_win, N_tok, C) → (B, C, D, H, W)
//
//   6. Connexion résiduelle + Layer Norm
//      Y = LayerNorm(X + Out)
//
// Paramètres :
//   channels     : C = nombre de canaux d'entrée/sortie
//   window_d/h/w : taille de la fenêtre d'attention (typiquement 4 ou 7)
//   num_heads    : nombre de têtes d'attention (C doit être divisible)
//   d_k          : dimension par tête = C / num_heads
//   use_residual : connexion résiduelle (true par défaut)
//   use_norm     : layer normalization (true par défaut)
//
// Interface Layer :
//   Entrée  : Tensor 5D (B, C, D, H, W)
//   Sortie  : Tensor 5D (B, C, D', H', W') avec D'=D, H'=H, W'=W
//   → dimensions spatiales préservées, s'insère partout dans CNN
//
// Compatibilité :
//   CNN::addLayer() — fonctionne directement
//   Après SparseConvAdapterLayer (sur Tensor dense produit par to_dense())
//   Après ConvLayer3D (pipeline dense)
//
// =============================================================================

class WindowAttention3DLayer : public Layer {
public:

    // ── Constructeur ──────────────────────────────────────────────────────────
    WindowAttention3DLayer(int  channels,
                           int  window_d    = 4,
                           int  window_h    = 4,
                           int  window_w    = 4,
                           int  num_heads   = 4,
                           bool use_residual = true,
                           bool use_norm     = true)
        : C_(channels),
          wd_(window_d), wh_(window_h), ww_(window_w),
          num_heads_(num_heads),
          use_residual_(use_residual),
          use_norm_(use_norm)
    {
        if (C_ % num_heads_ != 0)
            throw std::invalid_argument(
                "[WindowAttention3DLayer] channels (" + std::to_string(C_) +
                ") doit être divisible par num_heads (" +
                std::to_string(num_heads_) + ")");

        d_k_ = C_ / num_heads_;   // dimension par tête
        // scale : 1/sqrt(d_k) — correct pour le vrai multi-head (d_k = C/heads)
        scale_ = 1.0f / std::sqrt(static_cast<float>(d_k_));

        // Taille d'un token = C canaux
        // Projections : (C, C) pour Q, K, V et W_O
        W_Q_ = Eigen::MatrixXf(C_, C_);
        W_K_ = Eigen::MatrixXf(C_, C_);
        W_V_ = Eigen::MatrixXf(C_, C_);
        W_O_ = Eigen::MatrixXf(C_, C_);

        // Gradients
        dW_Q_ = Eigen::MatrixXf::Zero(C_, C_);
        dW_K_ = Eigen::MatrixXf::Zero(C_, C_);
        dW_V_ = Eigen::MatrixXf::Zero(C_, C_);
        dW_O_ = Eigen::MatrixXf::Zero(C_, C_);

        // Layer Norm : gamma et beta, taille C
        gamma_ = Eigen::VectorXf::Ones(C_);
        beta_  = Eigen::VectorXf::Zero(C_);
        d_gamma_ = Eigen::VectorXf::Zero(C_);
        d_beta_  = Eigen::VectorXf::Zero(C_);

        isTrainable = true;
        initializeWeights();
    }

    ~WindowAttention3DLayer() override = default;

    WindowAttention3DLayer(const WindowAttention3DLayer&)            = delete;
    WindowAttention3DLayer& operator=(const WindowAttention3DLayer&) = delete;

    // ── Initialisation Xavier ─────────────────────────────────────────────────
    void initializeWeights() {
        const float scale = std::sqrt(2.0f / (C_ + C_));
        std::mt19937 gen{std::random_device{}()};
        std::normal_distribution<float> dist(0.f, scale);

        auto fill = [&](Eigen::MatrixXf& M) {
            for (int i = 0; i < M.rows(); ++i)
                for (int j = 0; j < M.cols(); ++j)
                    M(i, j) = dist(gen);
        };
        fill(W_Q_); fill(W_K_); fill(W_V_); fill(W_O_);

        gamma_.setOnes();
        beta_.setZero();
    }

    // =========================================================================
    // FORWARD
    // =========================================================================
    //
    // Entrée  : (B, C, D, H, W)
    // Sortie  : (B, C, D, H, W)  — mêmes dimensions
    Tensor forward(const Tensor& input) override {

        if (input.ndim() != 5)
            throw std::runtime_error(
                "[WindowAttention3D] Attend Tensor 5D (B,C,D,H,W), reçu ndim="
                + std::to_string(input.ndim()));

        if (input.dim(1) != C_)
            throw std::runtime_error(
                "[WindowAttention3D] channels mismatch: attendu "
                + std::to_string(C_) + ", reçu " + std::to_string(input.dim(1)));

        // Dimensions seulement — les données ne sont pas relues en backward
        input_dims_ = { input.dim(0), input.dim(2), input.dim(3), input.dim(4) };

        const int B = input.dim(0);
        const int D = input.dim(2);
        const int H = input.dim(3);
        const int W = input.dim(4);

        // Dimensions des fenêtres (avec padding si nécessaire)
        const int nD = ceilDiv(D, wd_);   // nombre de fenêtres selon D
        const int nH = ceilDiv(H, wh_);   // nombre de fenêtres selon H
        const int nW = ceilDiv(W, ww_);   // nombre de fenêtres selon W
        const int N_win = nD * nH * nW;   // nombre total de fenêtres par image
        const int N_tok = wd_ * wh_ * ww_;// tokens par fenêtre

        // ── Étape 1 : Partition en fenêtres ───────────────────────────────────
        // tokens : (B, N_win, N_tok, C)
        //   B     : batch
        //   N_win : fenêtres par image
        //   N_tok : tokens par fenêtre (wd×wh×ww)
        //   C     : canaux (features par token)
        Eigen::MatrixXf tokens = partition(input, B, D, H, W, nD, nH, nW);
        // tokens.rows() = B * N_win * N_tok, tokens.cols() = C
        // Réorganisé comme : ligne i = token (b, win, tok)

        // Sauvegarde pour le backward
        tokens_cache_ = tokens;
        partition_info_ = {B, D, H, W, nD, nH, nW};

        // ── Étape 2 : Projections Q, K, V ────────────────────────────────────
        // tokens : (B*N_win*N_tok, C)
        // Q, K, V : (B*N_win*N_tok, C)
        Eigen::MatrixXf Q = tokens * W_Q_.transpose();   // (T, C)
        Eigen::MatrixXf K = tokens * W_K_.transpose();   // (T, C)
        Eigen::MatrixXf V = tokens * W_V_.transpose();   // (T, C)

        Q_cache_ = Q;
        K_cache_ = K;
        V_cache_ = V;

        // ── Étape 3 : Scores d'attention par fenêtre et par tête ─────────────
        // Pour chaque (fenêtre bw, tête h) :
        //   Q_h = Q[bw*N_tok:(bw+1)*N_tok, h*d_k:(h+1)*d_k]  (N_tok, d_k)
        //   S_h = Q_h · K_h^T / sqrt(d_k)                    (N_tok, N_tok)
        //   A_h = softmax(S_h)                                  (N_tok, N_tok)
        //   Out_h = A_h · V_h                                  (N_tok, d_k)
        //   attn_out[bw*N_tok:, h*d_k:] = Out_h
        const int BW = B * N_win;
        // A_cache_ taille BW * num_heads_ — un slot par (fenêtre, tête)
        A_cache_.assign(BW * num_heads_, Eigen::MatrixXf::Zero(N_tok, N_tok));

        Eigen::MatrixXf attn_out(BW * N_tok, C_);
        attn_out.setZero();

        for (int bw = 0; bw < BW; ++bw) {
            const Eigen::MatrixXf q_win = Q.middleRows(bw * N_tok, N_tok);  // (N_tok, C)
            const Eigen::MatrixXf k_win = K.middleRows(bw * N_tok, N_tok);
            const Eigen::MatrixXf v_win = V.middleRows(bw * N_tok, N_tok);

            for (int h = 0; h < num_heads_; ++h) {
                const int col0 = h * d_k_;

                // Sous-matrices de dimension d_k pour cette tête
                const Eigen::MatrixXf q_h = q_win.middleCols(col0, d_k_);  // (N_tok, d_k)
                const Eigen::MatrixXf k_h = k_win.middleCols(col0, d_k_);
                const Eigen::MatrixXf v_h = v_win.middleCols(col0, d_k_);

                // Score : (N_tok, N_tok) = (N_tok, d_k) * (d_k, N_tok)
                Eigen::MatrixXf S = (q_h * k_h.transpose()) * scale_;

                // Softmax numériquement stable
                Eigen::MatrixXf A = softmax(S);
                A_cache_[bw * num_heads_ + h] = A;

                // Agrégation : (N_tok, d_k) = (N_tok, N_tok) * (N_tok, d_k)
                attn_out.block(bw * N_tok, col0, N_tok, d_k_) = A * v_h;
            }
        }

        // ── Étape 4 : Projection de sortie W_O ───────────────────────────────
        // (B*N_win*N_tok, C) × (C, C) → (B*N_win*N_tok, C)
        attn_out_cache_ = attn_out;
        Eigen::MatrixXf out_proj = attn_out * W_O_.transpose();

        // ── Étape 5 : Reconstruction du Tensor ───────────────────────────────
        Tensor output = reconstruct(out_proj, B, D, H, W, nD, nH, nW);

        // ── Étape 6 : Connexion résiduelle + Layer Norm ───────────────────────
        if (use_residual_) {
            // Y = X + Attention(X)
            for (int i = 0; i < output.size(); ++i)
                output[i] += input[i];
        }

        if (use_norm_) {
            output = layerNorm(output, B);
        }

        return output;
    }

    // =========================================================================
    // BACKWARD
    // =========================================================================
    Tensor backward(const Tensor& grad_output) override {

        const int B   = partition_info_[0];
        const int D   = partition_info_[1];
        const int H   = partition_info_[2];
        const int W   = partition_info_[3];
        const int nD  = partition_info_[4];
        const int nH  = partition_info_[5];
        const int nW  = partition_info_[6];
        const int N_win = nD * nH * nW;
        const int N_tok = wd_ * wh_ * ww_;
        const int BW    = B * N_win;

        // ── 6b. Backward Layer Norm ───────────────────────────────────────────
        Tensor grad = grad_output;
        if (use_norm_) {
            grad = layerNormBackward(grad, B);
        }

        // ── 5b. Backward résiduelle ───────────────────────────────────────────
        // dL/dX_residual = dL/dY (le gradient passe directement via le skip)
        Tensor grad_residual = use_residual_ ? grad : Tensor(grad.shape());
        if (!use_residual_) grad_residual.setZero();

        // ── 5b. Backward reconstruction → partition ───────────────────────────
        // Reconvertit le gradient en format fenêtré (BW*N_tok, C)
        Eigen::MatrixXf d_out_proj = partition(grad, B, D, H, W, nD, nH, nW);

        // ── 4b. Backward projection W_O ───────────────────────────────────────
        // out_proj = attn_out * W_O^T
        // d_attn_out = d_out_proj * W_O
        // dW_O       = d_out_proj^T * attn_out / (B * N_win)  — normalisation correcte
        Eigen::MatrixXf d_attn_out = d_out_proj * W_O_;
        dW_O_ = (d_out_proj.transpose() * attn_out_cache_)
                / static_cast<float>(B * N_win);

        // ── 3b. Backward attention par fenêtre et par tête ─────────────────
        Eigen::MatrixXf d_Q = Eigen::MatrixXf::Zero(BW * N_tok, C_);
        Eigen::MatrixXf d_K = Eigen::MatrixXf::Zero(BW * N_tok, C_);
        Eigen::MatrixXf d_V = Eigen::MatrixXf::Zero(BW * N_tok, C_);

        for (int bw = 0; bw < BW; ++bw) {
            const Eigen::MatrixXf q_win = Q_cache_.middleRows(bw * N_tok, N_tok);  // (N_tok,C)
            const Eigen::MatrixXf k_win = K_cache_.middleRows(bw * N_tok, N_tok);
            const Eigen::MatrixXf v_win = V_cache_.middleRows(bw * N_tok, N_tok);
            const Eigen::MatrixXf dO_win = d_attn_out.middleRows(bw * N_tok, N_tok);

            Eigen::MatrixXf dq_win = Eigen::MatrixXf::Zero(N_tok, C_);
            Eigen::MatrixXf dk_win = Eigen::MatrixXf::Zero(N_tok, C_);
            Eigen::MatrixXf dv_win = Eigen::MatrixXf::Zero(N_tok, C_);

            for (int h = 0; h < num_heads_; ++h) {
                const int col0 = h * d_k_;
                const Eigen::MatrixXf& A   = A_cache_[bw * num_heads_ + h];
                const Eigen::MatrixXf q_h  = q_win.middleCols(col0, d_k_);
                const Eigen::MatrixXf k_h  = k_win.middleCols(col0, d_k_);
                const Eigen::MatrixXf v_h  = v_win.middleCols(col0, d_k_);
                const Eigen::MatrixXf dO_h = dO_win.middleCols(col0, d_k_);  // (N_tok, d_k)

                // dL/dV_h = A^T · dO_h    (N_tok, d_k)
                dv_win.middleCols(col0, d_k_) = A.transpose() * dO_h;

                // dL/dA = dO_h · V_h^T   (N_tok, N_tok)
                const Eigen::MatrixXf d_A = dO_h * v_h.transpose();

                // Backward softmax
                Eigen::MatrixXf d_S(N_tok, N_tok);
                for (int i = 0; i < N_tok; ++i) {
                    const float dot = (A.row(i).array() * d_A.row(i).array()).sum();
                    d_S.row(i) = A.row(i).array() * (d_A.row(i).array() - dot);
                }
                d_S *= scale_;

                // dL/dQ_h = d_S · K_h    (N_tok, d_k)
                dq_win.middleCols(col0, d_k_) = d_S * k_h;

                // dL/dK_h = d_S^T · Q_h  (N_tok, d_k)
                dk_win.middleCols(col0, d_k_) = d_S.transpose() * q_h;
            }

            d_Q.middleRows(bw * N_tok, N_tok) = dq_win;
            d_K.middleRows(bw * N_tok, N_tok) = dk_win;
            d_V.middleRows(bw * N_tok, N_tok) = dv_win;
        }

        // ── 2b. Backward projections W_Q, W_K, W_V ───────────────────────────
        // Q = tokens * W_Q^T → dL/dW_Q = d_Q^T * tokens / B
        //                       dL/dtokens += d_Q * W_Q
        // Normalisation par B*N_win — tous les contextes fenêtrés contribuent
        const float inv_BW = 1.f / static_cast<float>(BW);
        dW_Q_ = (d_Q.transpose() * tokens_cache_) * inv_BW;
        dW_K_ = (d_K.transpose() * tokens_cache_) * inv_BW;
        dW_V_ = (d_V.transpose() * tokens_cache_) * inv_BW;

        // Gradient vers les tokens d'entrée
        Eigen::MatrixXf d_tokens = d_Q * W_Q_
                                 + d_K * W_K_
                                 + d_V * W_V_;

        // ── 1b. Backward partition → Tensor ───────────────────────────────────
        Tensor grad_input = reconstruct(d_tokens, B, D, H, W, nD, nH, nW);

        // Ajout du gradient de la connexion résiduelle
        if (use_residual_) {
            for (int i = 0; i < grad_input.size(); ++i)
                grad_input[i] += grad_residual[i];
        }

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
        return "WindowAttention3D(C=" + std::to_string(C_)
             + " win=" + std::to_string(wd_) + "x"
             + std::to_string(wh_) + "x" + std::to_string(ww_)
             + " heads=" + std::to_string(num_heads_) + ")";
    }

    // ── Accesseurs ────────────────────────────────────────────────────────────
    Eigen::MatrixXf& getWQ() { return W_Q_; }
    Eigen::MatrixXf& getWK() { return W_K_; }
    Eigen::MatrixXf& getWV() { return W_V_; }
    Eigen::MatrixXf& getWO() { return W_O_; }

    // Nombre de paramètres total
    int numParams() const {
        return 4 * C_ * C_   // W_Q, W_K, W_V, W_O
             + 2 * C_;        // gamma, beta (LayerNorm)
    }

private:

    // ── Hyperparamètres ───────────────────────────────────────────────────────
    int   C_;
    int   wd_, wh_, ww_;
    int   num_heads_, d_k_;
    float scale_;
    bool  use_residual_, use_norm_;

    // ── Paramètres apprenables ────────────────────────────────────────────────
    Eigen::MatrixXf W_Q_, W_K_, W_V_, W_O_;   // (C, C) chacun
    Eigen::VectorXf gamma_, beta_;             // LayerNorm : (C)

    // ── Gradients ─────────────────────────────────────────────────────────────
    Eigen::MatrixXf dW_Q_, dW_K_, dW_V_, dW_O_;
    Eigen::VectorXf d_gamma_, d_beta_;

    // ── Cache backward ────────────────────────────────────────────────────────
    struct InputDims { int b, d, h, w; };   // dims seulement — pas de copie des données
    InputDims                   input_dims_{};
    Eigen::MatrixXf             tokens_cache_;   // (B*N_win*N_tok, C)
    Eigen::MatrixXf             Q_cache_, K_cache_, V_cache_;
    Eigen::MatrixXf             attn_out_cache_; // sortie avant W_O
    std::vector<Eigen::MatrixXf> A_cache_;       // matrices d'attention [BW]
    std::vector<int>            partition_info_; // {B,D,H,W,nD,nH,nW}

    // Cache LayerNorm
    Eigen::MatrixXf ln_x_norm_;   // (B*D*H*W, C) — x normalisé
    Eigen::VectorXf ln_mean_;     // (B*D*H*W)    — moyenne par token
    Eigen::VectorXf ln_var_;      // (B*D*H*W)    — variance par token
    int             ln_N_ = 0;    // nombre de tokens

    // ── Utilitaire : division entière supérieure ───────────────────────────────
    static int ceilDiv(int a, int b) { return (a + b - 1) / b; }

    // =========================================================================
    // PARTITION — Tensor (B,C,D,H,W) → matrice fenêtrée (B*N_win*N_tok, C)
    // =========================================================================
    // Découpe le volume en fenêtres non-chevauchantes de taille (wd,wh,ww).
    // Si D/H/W ne sont pas divisibles par wd/wh/ww, les tokens hors bornes
    // sont mis à zéro (zero-padding implicite).
    //
    // Ordre de parcours des tokens dans une fenêtre :
    //   (d_local, h_local, w_local) en RowMajor
    // Ordre des fenêtres :
    //   (b, nd, nh, nw) en RowMajor
    // =========================================================================
    Eigen::MatrixXf partition(const Tensor& t,
                               int B, int D, int H, int W,
                               int nD, int nH, int nW) const
    {
        const int N_win = nD * nH * nW;
        const int N_tok = wd_ * wh_ * ww_;

        Eigen::MatrixXf out = Eigen::MatrixXf::Zero(B * N_win * N_tok, C_);

        for (int b  = 0; b  < B;   ++b)
        for (int nd = 0; nd < nD;  ++nd)
        for (int nh = 0; nh < nH;  ++nh)
        for (int nw = 0; nw < nW;  ++nw) {
            const int win_idx = ((nd * nH + nh) * nW + nw);
            const int row_base = (b * N_win + win_idx) * N_tok;

            for (int td = 0; td < wd_; ++td)
            for (int th = 0; th < wh_; ++th)
            for (int tw = 0; tw < ww_; ++tw) {
                const int d = nd * wd_ + td;
                const int h = nh * wh_ + th;
                const int w = nw * ww_ + tw;

                const int tok_idx = (td * wh_ + th) * ww_ + tw;
                const int row = row_base + tok_idx;

                if (d < D && h < H && w < W) {
                    for (int c = 0; c < C_; ++c)
                        out(row, c) = t(b, c, d, h, w);
                }
                // sinon : zéro déjà initialisé
            }
        }
        return out;
    }

    // =========================================================================
    // RECONSTRUCT — matrice fenêtrée (B*N_win*N_tok, C) → Tensor (B,C,D,H,W)
    // =========================================================================
    // Opération inverse de partition().
    // Les positions de padding (hors bornes) sont ignorées.
    // En cas de chevauchement (absent ici — fenêtres non-chevauchantes),
    // on accumulerait ; ici on affecte directement.
    // =========================================================================
    Tensor reconstruct(const Eigen::MatrixXf& mat,
                       int B, int D, int H, int W,
                       int nD, int nH, int nW) const
    {
        const int N_win = nD * nH * nW;
        const int N_tok = wd_ * wh_ * ww_;

        Tensor out(B, C_, D, H, W);
        out.setZero();

        for (int b  = 0; b  < B;   ++b)
        for (int nd = 0; nd < nD;  ++nd)
        for (int nh = 0; nh < nH;  ++nh)
        for (int nw = 0; nw < nW;  ++nw) {
            const int win_idx  = ((nd * nH + nh) * nW + nw);
            const int row_base = (b * N_win + win_idx) * N_tok;

            for (int td = 0; td < wd_; ++td)
            for (int th = 0; th < wh_; ++th)
            for (int tw = 0; tw < ww_; ++tw) {
                const int d = nd * wd_ + td;
                const int h = nh * wh_ + th;
                const int w = nw * ww_ + tw;

                if (d >= D || h >= H || w >= W) continue;

                const int tok_idx = (td * wh_ + th) * ww_ + tw;
                const int row     = row_base + tok_idx;

                for (int c = 0; c < C_; ++c)
                    out(b, c, d, h, w) = mat(row, c);
            }
        }
        return out;
    }

    // =========================================================================
    // SOFTMAX par ligne — stabilisé numériquement
    // =========================================================================
    static Eigen::MatrixXf softmax(const Eigen::MatrixXf& S) {
        Eigen::MatrixXf A(S.rows(), S.cols());
        for (int i = 0; i < S.rows(); ++i) {
            const float max_val = S.row(i).maxCoeff();
            Eigen::RowVectorXf e = (S.row(i).array() - max_val).exp();
            A.row(i) = e / e.sum();
        }
        return A;
    }

    // =========================================================================
    // LAYER NORMALIZATION
    // =========================================================================
    // Normalise chaque token (vecteur de C canaux) indépendamment.
    // Y[i] = gamma ⊙ (X[i] - mean[i]) / sqrt(var[i] + eps) + beta
    //
    // Appliqué après aplatissement en (N_tokens_total, C) puis reconstruction.
    // =========================================================================
    Tensor layerNorm(const Tensor& t, int B) {
        const float eps = 1e-5f;
        const int D = t.dim(2), H = t.dim(3), W = t.dim(4);
        const int N = B * D * H * W;   // nombre total de tokens

        // Aplatir en (N, C)
        Eigen::MatrixXf X(N, C_);
        int idx = 0;
        for (int b = 0; b < B; ++b)
        for (int d = 0; d < D; ++d)
        for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            for (int c = 0; c < C_; ++c)
                X(idx, c) = t(b, c, d, h, w);
            ++idx;
        }

        // Statistiques par token
        ln_mean_ = X.rowwise().mean();                   // (N)
        Eigen::MatrixXf X_centered = X.colwise() - ln_mean_;
        ln_var_  = X_centered.array().square().rowwise().mean(); // (N)

        // Normalisation
        ln_x_norm_.resize(N, C_);
        for (int i = 0; i < N; ++i) {
            const float std_inv = 1.f / std::sqrt(ln_var_[i] + eps);
            for (int c = 0; c < C_; ++c)
                ln_x_norm_(i, c) = X_centered(i, c) * std_inv;
        }
        ln_N_ = N;

        // Mise à l'échelle et décalage : gamma ⊙ x_norm + beta
        Eigen::MatrixXf Y = (ln_x_norm_.array().rowwise() *
                             gamma_.transpose().array())
                            .rowwise() + beta_.transpose().array();

        // Reconstruction en Tensor
        Tensor out(B, C_, D, H, W);
        idx = 0;
        for (int b = 0; b < B; ++b)
        for (int d = 0; d < D; ++d)
        for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            for (int c = 0; c < C_; ++c)
                out(b, c, d, h, w) = Y(idx, c);
            ++idx;
        }
        return out;
    }

    // =========================================================================
    // BACKWARD Layer Normalization
    // =========================================================================
    Tensor layerNormBackward(const Tensor& grad, int B) {
        const float eps = 1e-5f;
        const int D = grad.dim(2), H = grad.dim(3), W = grad.dim(4);

        // Aplatir gradient en (N, C)
        Eigen::MatrixXf dY(ln_N_, C_);
        int idx = 0;
        for (int b = 0; b < B; ++b)
        for (int d = 0; d < D; ++d)
        for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            for (int c = 0; c < C_; ++c)
                dY(idx, c) = grad(b, c, d, h, w);
            ++idx;
        }

        // Gradients de gamma et beta
        // .matrix() requis : Eigen interdit += entre VectorXf et ArrayXf
        d_gamma_ += ((dY.array() * ln_x_norm_.array()).colwise().sum()
                  / static_cast<float>(B)).matrix().transpose();
        d_beta_  += (dY.colwise().sum()
                  / static_cast<float>(B)).transpose();

        // Gradient vers l'entrée (formule standard backward LayerNorm)
        Eigen::MatrixXf dX(ln_N_, C_);
        for (int i = 0; i < ln_N_; ++i) {
            const float std_inv = 1.f / std::sqrt(ln_var_[i] + eps);
            // dL/dx = (1/C) * std_inv * (C*dY_hat - sum(dY_hat) - x_norm*sum(dY_hat*x_norm))
            // où dY_hat = dY ⊙ gamma
            Eigen::RowVectorXf dY_hat = dY.row(i).array()
                                        * gamma_.transpose().array();
            const float s1 = dY_hat.sum();
            const float s2 = (dY_hat.array() * ln_x_norm_.row(i).array()).sum();
            for (int c = 0; c < C_; ++c)
                dX(i, c) = std_inv / C_
                          * (C_ * dY_hat(c) - s1 - ln_x_norm_(i,c) * s2);
        }

        // Reconstruction en Tensor
        Tensor grad_input(B, C_, D, H, W);
        idx = 0;
        for (int b = 0; b < B; ++b)
        for (int d = 0; d < D; ++d)
        for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            for (int c = 0; c < C_; ++c)
                grad_input(b, c, d, h, w) = dX(idx, c);
            ++idx;
        }
        return grad_input;
    }
};