



#pragma once
#include "Layer.hpp"
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include "ModelSerializer.hpp"

// =============================================================================
// WindowAttention3DLayer — Attention par Fenêtres Volumétriques 3D
// =============================================================================
//
// Principe (inspiré de Swin Transformer 3D / Video Swin Transformer) :
//
//   Le volume est découpé en fenêtres locales non-chevauchantes de taille
//   (wd, wh, ww). L'attention est calculée indépendamment dans chaque fenêtre.
//
//   Pour un volume (B, C, 14, 14, 14) avec fenêtres 4×4×4  [Attn-1] :
//     N_tokens_par_fenetre = 4×4×4 = 64
//     N_fenetres = ceil(14/4)^3 = 4^3 = 64
//     Matrice d'attention : 64×64 par fenêtre
//     Gain vs attention globale 14³ : facteur ≈ 43×
//
//   Pour un volume (B, C, 7, 7, 7) avec fenêtre 7×7×7  [Attn-2] :
//     N_tokens_par_fenetre = 7×7×7 = 343
//     N_fenetres = ceil(7/7)^3 = 1  — attention globale à cette résolution
//     Matrice d'attention : 343×343
//
// Positional Encoding appris (Learned Absolute Position Embedding) :
//
//   Chaque position locale (td, th, tw) dans une fenêtre reçoit un vecteur
//   appris P ∈ R^C. Ce vecteur est ajouté aux tokens après partition et
//   avant les projections Q, K, V :
//
//     X'_i = X_i + P_i,   i ∈ {0, …, N_tok - 1}
//
//   pos_embed_ : (N_tok, C) — un vecteur par position locale dans la fenêtre
//   Initialisé avec une distribution normale N(0, 0.02) (convention ViT).
//   Partagé entre toutes les fenêtres et tous les éléments du batch.
//
// Algorithme (forward) :
//
//   1. Partition  : (B, C, D, H, W) → (B*N_win, N_tok, C)
//   2. + pos_embed_ broadcast sur (B*N_win, N_tok, C)
//   3. Projections Q = X'·W_Q,  K = X'·W_K,  V = X'·W_V
//   4. S = Q·K^T / sqrt(d_k),  A = softmax(S)
//   5. Out = A·V·W_O
//   6. Reconstruction + résiduelle + LayerNorm
//
// =============================================================================

class WindowAttention3DLayer : public Layer {
public:

    // ── Constructeur ──────────────────────────────────────────────────────────
    WindowAttention3DLayer(int  channels,
                           int  window_d     = 4,
                           int  window_h     = 4,
                           int  window_w     = 4,
                           int  num_heads    = 4,
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

        d_k_   = C_ / num_heads_;
        scale_ = 1.0f / std::sqrt(static_cast<float>(d_k_));
        N_tok_ = wd_ * wh_ * ww_;   // tokens par fenêtre — fixe pour cette instance

        // Projections d'attention : (C, C)
        W_Q_ = Eigen::MatrixXf(C_, C_);
        W_K_ = Eigen::MatrixXf(C_, C_);
        W_V_ = Eigen::MatrixXf(C_, C_);
        W_O_ = Eigen::MatrixXf(C_, C_);
        dW_Q_ = Eigen::MatrixXf::Zero(C_, C_);
        dW_K_ = Eigen::MatrixXf::Zero(C_, C_);
        dW_V_ = Eigen::MatrixXf::Zero(C_, C_);
        dW_O_ = Eigen::MatrixXf::Zero(C_, C_);

        // Positional encoding appris : (N_tok, C)
        // Un vecteur par position locale dans la fenêtre, partagé entre fenêtres
        pos_embed_  = Eigen::MatrixXf::Zero(N_tok_, C_);
        d_pos_embed_ = Eigen::MatrixXf::Zero(N_tok_, C_);

        // LayerNorm
        gamma_   = Eigen::VectorXf::Ones(C_);
        beta_    = Eigen::VectorXf::Zero(C_);
        d_gamma_ = Eigen::VectorXf::Zero(C_);
        d_beta_  = Eigen::VectorXf::Zero(C_);

        isTrainable = true;
        initializeWeights();
    }

    ~WindowAttention3DLayer() override = default;

    WindowAttention3DLayer(const WindowAttention3DLayer&)            = delete;
    WindowAttention3DLayer& operator=(const WindowAttention3DLayer&) = delete;

    // ── Initialisation des poids ───────────────────────────────────────────────
    // W_Q/K/V/O : Xavier uniforme — convention standard pour les projections
    // pos_embed_ : N(0, 0.02) — convention ViT / Swin Transformer
    void initializeWeights() {
        // Xavier pour les projections d'attention
        const float xavier_scale = std::sqrt(2.0f / (C_ + C_));
        std::mt19937 gen{std::random_device{}()};
        std::normal_distribution<float> xavier_dist(0.f, xavier_scale);

        auto fill_matrix = [&](Eigen::MatrixXf& M) {
            for (int i = 0; i < M.rows(); ++i)
                for (int j = 0; j < M.cols(); ++j)
                    M(i, j) = xavier_dist(gen);
        };
        fill_matrix(W_Q_); fill_matrix(W_K_);
        fill_matrix(W_V_); fill_matrix(W_O_);

        // N(0, 0.02) pour le positional encoding — petite variance intentionnelle
        // pour ne pas dominer les features au début de l'entraînement
        std::normal_distribution<float> pe_dist(0.f, 0.02f);
        for (int i = 0; i < pos_embed_.rows(); ++i)
            for (int j = 0; j < pos_embed_.cols(); ++j)
                pos_embed_(i, j) = pe_dist(gen);

        gamma_.setOnes();
        beta_.setZero();
    }

    // =========================================================================
    // FORWARD  (B, C, D, H, W) → (B, C, D, H, W)
    // =========================================================================
    Tensor forward(const Tensor& input) override {

        if (input.ndim() != 5)
            throw std::runtime_error(
                "[WindowAttention3D] Attend Tensor 5D (B,C,D,H,W), reçu ndim="
                + std::to_string(input.ndim()));
        if (input.dim(1) != C_)
            throw std::runtime_error(
                "[WindowAttention3D] channels mismatch: attendu "
                + std::to_string(C_) + ", reçu " + std::to_string(input.dim(1)));

        const int B = input.dim(0);
        const int D = input.dim(2);
        const int H = input.dim(3);
        const int W = input.dim(4);

        const int nD    = ceilDiv(D, wd_);
        const int nH    = ceilDiv(H, wh_);
        const int nW    = ceilDiv(W, ww_);
        const int N_win = nD * nH * nW;
        const int BW    = B * N_win;

        partition_info_ = {B, D, H, W, nD, nH, nW};

        // ── Étape 1 : Partition ───────────────────────────────────────────────
        // tokens_raw : (BW * N_tok_, C) — features brutes sans position
        Eigen::MatrixXf tokens_raw = partition(input, B, D, H, W, nD, nH, nW);

        // ── Étape 2 : Ajout du positional encoding ────────────────────────────
        // pos_embed_ : (N_tok_, C) — broadcasté sur toutes les fenêtres et le batch
        // tokens[bw * N_tok_ + t, :] += pos_embed_[t, :]
        //
        // Implémentation : on répète pos_embed_ BW fois pour former une matrice
        // (BW * N_tok_, C), puis addition terme à terme.
        Eigen::MatrixXf tokens = tokens_raw;
        for (int bw = 0; bw < BW; ++bw)
            tokens.middleRows(bw * N_tok_, N_tok_) += pos_embed_;

        // Sauvegarde pour le backward
        tokens_cache_  = tokens;       // après ajout PE — utilisé pour dW_Q/K/V
        tokens_raw_cache_ = tokens_raw; // avant PE — utilisé pour d_pos_embed_

        // ── Étape 3 : Projections Q, K, V ────────────────────────────────────
        Eigen::MatrixXf Q = tokens * W_Q_.transpose();
        Eigen::MatrixXf K = tokens * W_K_.transpose();
        Eigen::MatrixXf V = tokens * W_V_.transpose();
        Q_cache_ = Q; K_cache_ = K; V_cache_ = V;

        // ── Étape 4 : Scores d'attention par fenêtre et par tête ─────────────
        A_cache_.assign(BW * num_heads_, Eigen::MatrixXf::Zero(N_tok_, N_tok_));
        Eigen::MatrixXf attn_out(BW * N_tok_, C_);
        attn_out.setZero();

        for (int bw = 0; bw < BW; ++bw) {
            const Eigen::MatrixXf q_win = Q.middleRows(bw * N_tok_, N_tok_);
            const Eigen::MatrixXf k_win = K.middleRows(bw * N_tok_, N_tok_);
            const Eigen::MatrixXf v_win = V.middleRows(bw * N_tok_, N_tok_);

            for (int h = 0; h < num_heads_; ++h) {
                const int col0 = h * d_k_;
                const Eigen::MatrixXf q_h = q_win.middleCols(col0, d_k_);
                const Eigen::MatrixXf k_h = k_win.middleCols(col0, d_k_);
                const Eigen::MatrixXf v_h = v_win.middleCols(col0, d_k_);

                Eigen::MatrixXf S = (q_h * k_h.transpose()) * scale_;
                Eigen::MatrixXf A = softmax(S);
                A_cache_[bw * num_heads_ + h] = A;
                attn_out.block(bw * N_tok_, col0, N_tok_, d_k_) = A * v_h;
            }
        }

        // ── Étape 5 : Projection de sortie W_O ───────────────────────────────
        attn_out_cache_ = attn_out;
        Eigen::MatrixXf out_proj = attn_out * W_O_.transpose();

        // ── Étape 6 : Reconstruction ──────────────────────────────────────────
        Tensor output = reconstruct(out_proj, B, D, H, W, nD, nH, nW);

        // ── Étape 7 : Résiduelle + LayerNorm ─────────────────────────────────
        if (use_residual_)
            for (int i = 0; i < output.size(); ++i)
                output[i] += input[i];

        if (use_norm_)
            output = layerNorm(output, B);

        return output;
    }

    // =========================================================================
    // BACKWARD
    // =========================================================================
    Tensor backward(const Tensor& grad_output) override {

        const int B     = partition_info_[0];
        const int D     = partition_info_[1];
        const int H     = partition_info_[2];
        const int W     = partition_info_[3];
        const int nD    = partition_info_[4];
        const int nH    = partition_info_[5];
        const int nW    = partition_info_[6];
        const int N_win = nD * nH * nW;
        const int BW    = B * N_win;

        // ── 7b. Backward LayerNorm ────────────────────────────────────────────
        Tensor grad = grad_output;
        if (use_norm_)
            grad = layerNormBackward(grad, B);

        // ── 7b. Backward résiduelle ───────────────────────────────────────────
        Tensor grad_residual(grad.shape());
        grad_residual.setZero();
        if (use_residual_)
            grad_residual = grad;

        // ── 6b. Backward reconstruction → partition ───────────────────────────
        Eigen::MatrixXf d_out_proj = partition(grad, B, D, H, W, nD, nH, nW);

        // ── 5b. Backward W_O ──────────────────────────────────────────────────
        Eigen::MatrixXf d_attn_out = d_out_proj * W_O_;
        dW_O_ = (d_out_proj.transpose() * attn_out_cache_)
                / static_cast<float>(BW);

        // ── 4b. Backward attention ────────────────────────────────────────────
        Eigen::MatrixXf d_Q = Eigen::MatrixXf::Zero(BW * N_tok_, C_);
        Eigen::MatrixXf d_K = Eigen::MatrixXf::Zero(BW * N_tok_, C_);
        Eigen::MatrixXf d_V = Eigen::MatrixXf::Zero(BW * N_tok_, C_);

        for (int bw = 0; bw < BW; ++bw) {
            const Eigen::MatrixXf q_win  = Q_cache_.middleRows(bw * N_tok_, N_tok_);
            const Eigen::MatrixXf k_win  = K_cache_.middleRows(bw * N_tok_, N_tok_);
            const Eigen::MatrixXf v_win  = V_cache_.middleRows(bw * N_tok_, N_tok_);
            const Eigen::MatrixXf dO_win = d_attn_out.middleRows(bw * N_tok_, N_tok_);

            Eigen::MatrixXf dq_win = Eigen::MatrixXf::Zero(N_tok_, C_);
            Eigen::MatrixXf dk_win = Eigen::MatrixXf::Zero(N_tok_, C_);
            Eigen::MatrixXf dv_win = Eigen::MatrixXf::Zero(N_tok_, C_);

            for (int h = 0; h < num_heads_; ++h) {
                const int col0 = h * d_k_;
                const Eigen::MatrixXf& A  = A_cache_[bw * num_heads_ + h];
                const Eigen::MatrixXf q_h = q_win.middleCols(col0, d_k_);
                const Eigen::MatrixXf k_h = k_win.middleCols(col0, d_k_);
                const Eigen::MatrixXf v_h = v_win.middleCols(col0, d_k_);
                const Eigen::MatrixXf dO_h = dO_win.middleCols(col0, d_k_);

                dv_win.middleCols(col0, d_k_) = A.transpose() * dO_h;

                const Eigen::MatrixXf d_A = dO_h * v_h.transpose();
                Eigen::MatrixXf d_S(N_tok_, N_tok_);
                for (int i = 0; i < N_tok_; ++i) {
                    const float dot = (A.row(i).array() * d_A.row(i).array()).sum();
                    d_S.row(i) = A.row(i).array() * (d_A.row(i).array() - dot);
                }
                d_S *= scale_;

                dq_win.middleCols(col0, d_k_) = d_S * k_h;
                dk_win.middleCols(col0, d_k_) = d_S.transpose() * q_h;
            }

            d_Q.middleRows(bw * N_tok_, N_tok_) = dq_win;
            d_K.middleRows(bw * N_tok_, N_tok_) = dk_win;
            d_V.middleRows(bw * N_tok_, N_tok_) = dv_win;
        }

        // ── 3b. Backward W_Q, W_K, W_V ───────────────────────────────────────
        const float inv_BW = 1.f / static_cast<float>(BW);
        dW_Q_ = (d_Q.transpose() * tokens_cache_) * inv_BW;
        dW_K_ = (d_K.transpose() * tokens_cache_) * inv_BW;
        dW_V_ = (d_V.transpose() * tokens_cache_) * inv_BW;

        // Gradient vers les tokens (après PE)
        Eigen::MatrixXf d_tokens = d_Q * W_Q_ + d_K * W_K_ + d_V * W_V_;

        // ── 2b. Backward positional encoding ─────────────────────────────────
        // pos_embed_ est broadcasté sur BW fenêtres → gradient = somme sur BW
        // d_pos_embed_[t, :] = sum_{bw} d_tokens[bw*N_tok_ + t, :]
        d_pos_embed_.setZero();
        for (int bw = 0; bw < BW; ++bw)
            d_pos_embed_ += d_tokens.middleRows(bw * N_tok_, N_tok_);
        d_pos_embed_ /= static_cast<float>(BW);  // normalisation cohérente avec dW

        // Le gradient vers tokens_raw est identique à d_tokens
        // (l'addition PE est une opération transparente pour le gradient d'entrée)

        // ── 1b. Backward partition → Tensor ───────────────────────────────────
        Tensor grad_input = reconstruct(d_tokens, B, D, H, W, nD, nH, nW);

        // Ajout du gradient résiduel
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
        optimizer.updateWeights(pos_embed_, d_pos_embed_);   // PE via Adam
        optimizer.updateBias(gamma_, d_gamma_);
        optimizer.updateBias(beta_,  d_beta_);

        dW_Q_.setZero(); dW_K_.setZero();
        dW_V_.setZero(); dW_O_.setZero();
        d_pos_embed_.setZero();
        d_gamma_.setZero(); d_beta_.setZero();
    }

    std::string getName() const override {
        return "WindowAttention3D(C=" + std::to_string(C_)
             + " win=" + std::to_string(wd_) + "x"
             + std::to_string(wh_) + "x" + std::to_string(ww_)
             + " heads=" + std::to_string(num_heads_)
             + " PE=learned)";
    }

    void saveParameters(boost::archive::binary_oarchive& archive) const override {
        archive << W_Q_ << W_K_ << W_V_ << W_O_;
        archive << pos_embed_;
        archive << gamma_ << beta_;
    }

    void loadParameters(boost::archive::binary_iarchive& archive) override {
        archive >> W_Q_ >> W_K_ >> W_V_ >> W_O_;
        archive >> pos_embed_;
        archive >> gamma_ >> beta_;
    }

    // Nombre de paramètres total
    int numParams() const {
        return 4 * C_ * C_       // W_Q, W_K, W_V, W_O
             + N_tok_ * C_       // pos_embed_  ← nouveau
             + 2 * C_;           // gamma, beta
    }

    Eigen::MatrixXf& getWQ() { return W_Q_; }
    Eigen::MatrixXf& getWK() { return W_K_; }
    Eigen::MatrixXf& getWV() { return W_V_; }
    Eigen::MatrixXf& getWO() { return W_O_; }
    Eigen::MatrixXf& getPosEmbed() { return pos_embed_; }

private:

    // ── Hyperparamètres ───────────────────────────────────────────────────────
    int   C_, wd_, wh_, ww_, num_heads_, d_k_, N_tok_;
    float scale_;
    bool  use_residual_, use_norm_;

    // ── Paramètres apprenables ────────────────────────────────────────────────
    Eigen::MatrixXf W_Q_, W_K_, W_V_, W_O_;     // (C, C)
    Eigen::MatrixXf pos_embed_;                  // (N_tok_, C) — PE appris
    Eigen::VectorXf gamma_, beta_;               // LayerNorm (C)

    // ── Gradients ─────────────────────────────────────────────────────────────
    Eigen::MatrixXf dW_Q_, dW_K_, dW_V_, dW_O_;
    Eigen::MatrixXf d_pos_embed_;                // (N_tok_, C)
    Eigen::VectorXf d_gamma_, d_beta_;

    // ── Cache backward ────────────────────────────────────────────────────────
    Eigen::MatrixXf              tokens_cache_;      // tokens après PE
    Eigen::MatrixXf              tokens_raw_cache_;  // tokens avant PE
    Eigen::MatrixXf              Q_cache_, K_cache_, V_cache_;
    Eigen::MatrixXf              attn_out_cache_;
    std::vector<Eigen::MatrixXf> A_cache_;
    std::vector<int>             partition_info_;    // {B,D,H,W,nD,nH,nW}

    // Cache LayerNorm
    Eigen::MatrixXf ln_x_norm_;
    Eigen::VectorXf ln_mean_, ln_var_;
    int             ln_N_ = 0;

    // ── Utilitaires ───────────────────────────────────────────────────────────
    static int ceilDiv(int a, int b) { return (a + b - 1) / b; }

    // =========================================================================
    // PARTITION — Tensor (B,C,D,H,W) → (B*N_win*N_tok, C)
    // =========================================================================
    Eigen::MatrixXf partition(const Tensor& t,
                               int B, int D, int H, int W,
                               int nD, int nH, int nW) const
    {
        const int N_win = nD * nH * nW;
        Eigen::MatrixXf out = Eigen::MatrixXf::Zero(B * N_win * N_tok_, C_);

        for (int b  = 0; b  < B;   ++b)
        for (int nd = 0; nd < nD;  ++nd)
        for (int nh = 0; nh < nH;  ++nh)
        for (int nw = 0; nw < nW;  ++nw) {
            const int win_idx  = (nd * nH + nh) * nW + nw;
            const int row_base = (b * N_win + win_idx) * N_tok_;

            for (int td = 0; td < wd_; ++td)
            for (int th = 0; th < wh_; ++th)
            for (int tw = 0; tw < ww_; ++tw) {
                const int d = nd * wd_ + td;
                const int h = nh * wh_ + th;
                const int w = nw * ww_ + tw;
                const int tok_idx = (td * wh_ + th) * ww_ + tw;
                const int row = row_base + tok_idx;

                if (d < D && h < H && w < W)
                    for (int c = 0; c < C_; ++c)
                        out(row, c) = t(b, c, d, h, w);
            }
        }
        return out;
    }

    // =========================================================================
    // RECONSTRUCT — (B*N_win*N_tok, C) → Tensor (B,C,D,H,W)
    // =========================================================================
    Tensor reconstruct(const Eigen::MatrixXf& mat,
                       int B, int D, int H, int W,
                       int nD, int nH, int nW) const
    {
        const int N_win = nD * nH * nW;
        Tensor out(B, C_, D, H, W);
        out.setZero();

        for (int b  = 0; b  < B;   ++b)
        for (int nd = 0; nd < nD;  ++nd)
        for (int nh = 0; nh < nH;  ++nh)
        for (int nw = 0; nw < nW;  ++nw) {
            const int win_idx  = (nd * nH + nh) * nW + nw;
            const int row_base = (b * N_win + win_idx) * N_tok_;

            for (int td = 0; td < wd_; ++td)
            for (int th = 0; th < wh_; ++th)
            for (int tw = 0; tw < ww_; ++tw) {
                const int d = nd * wd_ + td;
                const int h = nh * wh_ + th;
                const int w = nw * ww_ + tw;
                if (d >= D || h >= H || w >= W) continue;

                const int tok_idx = (td * wh_ + th) * ww_ + tw;
                const int row = row_base + tok_idx;
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
    // LAYER NORMALIZATION — forward
    // Normalise chaque token (vecteur C) indépendamment.
    // =========================================================================
    Tensor layerNorm(const Tensor& t, int B) {
        const float eps = 1e-5f;
        const int D = t.dim(2), H = t.dim(3), W = t.dim(4);
        const int N = B * D * H * W;

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

        ln_mean_ = X.rowwise().mean();
        Eigen::MatrixXf X_c = X.colwise() - ln_mean_;
        ln_var_  = X_c.array().square().rowwise().mean();
        ln_N_    = N;

        ln_x_norm_.resize(N, C_);
        for (int i = 0; i < N; ++i) {
            const float std_inv = 1.f / std::sqrt(ln_var_[i] + eps);
            for (int c = 0; c < C_; ++c)
                ln_x_norm_(i, c) = X_c(i, c) * std_inv;
        }

        Eigen::MatrixXf Y = (ln_x_norm_.array().rowwise() *
                             gamma_.transpose().array())
                            .rowwise() + beta_.transpose().array();

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
    // LAYER NORMALIZATION — backward
    // Correction : normalisation par ln_N_ (= B*D*H*W) et non par B seul
    // =========================================================================
    Tensor layerNormBackward(const Tensor& grad, int B) {
        const float eps = 1e-5f;
        const int D = grad.dim(2), H = grad.dim(3), W = grad.dim(4);

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

        // Gradients gamma et beta — normalisés par ln_N_ (correction du bug)
        d_gamma_ += ((dY.array() * ln_x_norm_.array()).colwise().sum()
                  / static_cast<float>(ln_N_)).matrix().transpose();
        d_beta_  += (dY.colwise().sum()
                  / static_cast<float>(ln_N_)).transpose();

        // Gradient d'entrée
        Eigen::MatrixXf dX(ln_N_, C_);
        for (int i = 0; i < ln_N_; ++i) {
            const float std_inv = 1.f / std::sqrt(ln_var_[i] + eps);
            Eigen::RowVectorXf dY_hat = dY.row(i).array()
                                      * gamma_.transpose().array();
            const float s1 = dY_hat.sum();
            const float s2 = (dY_hat.array() * ln_x_norm_.row(i).array()).sum();
            for (int c = 0; c < C_; ++c)
                dX(i, c) = std_inv / C_
                          * (C_ * dY_hat(c) - s1 - ln_x_norm_(i, c) * s2);
        }

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