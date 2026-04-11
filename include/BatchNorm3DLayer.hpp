// #pragma once
// #include <vector>
// #include <cmath>
// #include <stdexcept>
// #include <numeric>
// #include "Layer.hpp"
// #include "Tensor.hpp"
// #include "ModelSerializer.hpp"

// // ---------------------------------------------------------------------------
// //  BatchNorm3D — Batch Normalisation pour volumes 3D
// //
// //  Entrée  : Tensor (N, C, D, H, W)
// //  Sortie  : Tensor (N, C, D, H, W)   — même shape
// //
// //  Pour chaque canal c :
// //      mu_c    = mean  sur (N, D, H, W)
// //      var_c   = var   sur (N, D, H, W)
// //      x_hat   = (x - mu_c) / sqrt(var_c + eps)
// //      y       = gamma_c * x_hat + beta_c
// //
// //  En inférence on utilise les running stats (EMA).
// // ---------------------------------------------------------------------------
// class BatchNorm3D : public Layer {
// public:
//     // -----------------------------------------------------------------------
//     //  Constructeur
//     //  num_features : nombre de canaux C
//     //  eps          : stabilité numérique (défaut 1e-5)
//     //  momentum     : coefficient EMA pour les running stats (défaut 0.1)
//     // -----------------------------------------------------------------------
//     explicit BatchNorm3D(int num_features,
//                          float eps      = 1e-5f,
//                          float momentum = 0.1f)
//         : num_features_(num_features),
//           eps_(eps),
//           momentum_(momentum),
//           training_(true)
//     {
//         // Paramètres apprenables
//         gamma_.assign(num_features, 1.0f);   // scale  — initialisé à 1
//         beta_.assign(num_features,  0.0f);   // shift  — initialisé à 0

//         // Gradients des paramètres
//         d_gamma_.assign(num_features, 0.0f);
//         d_beta_.assign(num_features,  0.0f);

//         // Running statistics (inférence)
//         running_mean_.assign(num_features, 0.0f);
//         running_var_.assign(num_features,  1.0f);
        
//         // Cette couche est entraînable
//         isTrainable = true;
//     }

//     // -----------------------------------------------------------------------
//     //  Implémentation des méthodes virtuelles pures de Layer
//     // -----------------------------------------------------------------------
    
//     std::string getName() const override {
//         return "BatchNorm3D";
//     }

//     void saveParameters(boost::archive::binary_oarchive& archive) const override {
//         archive << gamma_;
//         archive << beta_;
//         archive << running_mean_;
//         archive << running_var_;
//     }

//     void loadParameters(boost::archive::binary_iarchive& archive) override {
//         archive >> gamma_;
//         archive >> beta_;
//         archive >> running_mean_;
//         archive >> running_var_;
//     }

//     // Forward pass
//     Tensor forward(const Tensor& input) override {
//         // Vérifications
//         if (input.ndim() != 5) {
//             throw std::invalid_argument(
//                 "BatchNorm3D: expected 5D tensor (B,C,D,H,W), got " +
//                 std::to_string(input.ndim()) + "D");
//         }

//         const int B = input.dim(0);
//         const int C = input.dim(1);
//         const int D = input.dim(2);
//         const int H = input.dim(3);
//         const int W = input.dim(4);
        
//         if (C != num_features_) {
//             throw std::invalid_argument(
//                 "BatchNorm3D: expected " + std::to_string(num_features_) +
//                 " channels, got " + std::to_string(C));
//         }

//         const int spatial = D * H * W;           // éléments par (sample, canal)
//         const int M = B * spatial;                // nb de valeurs moyennées par canal

//         // Sauvegarde pour le backward
//         input_ = input;
//         shape_ = input.shape();
        
//         // Création du tenseur de sortie
//         Tensor output(B, C, D, H, W);
        
//         // Buffers temporaires
//         x_hat_.resize(B * C * D * H * W);
//         mean_.resize(C);
//         inv_std_.resize(C);

//         for (int c = 0; c < C; ++c) {
//             // --- Calcul de la moyenne du canal c ---
//             float mu = 0.0f;
//             for (int b = 0; b < B; ++b) {
//                 for (int d = 0; d < D; ++d) {
//                     for (int h = 0; h < H; ++h) {
//                         for (int w = 0; w < W; ++w) {
//                             mu += input(b, c, d, h, w);
//                         }
//                     }
//                 }
//             }
//             mu /= static_cast<float>(M);

//             // --- Calcul de la variance ---
//             float var = 0.0f;
//             for (int b = 0; b < B; ++b) {
//                 for (int d = 0; d < D; ++d) {
//                     for (int h = 0; h < H; ++h) {
//                         for (int w = 0; w < W; ++w) {
//                             float diff = input(b, c, d, h, w) - mu;
//                             var += diff * diff;
//                         }
//                     }
//                 }
//             }
//             var /= static_cast<float>(M);

//             const float inv_std_val = 1.0f / std::sqrt(var + eps_);

//             if (training_) {
//                 mean_[c] = mu;
//                 inv_std_[c] = inv_std_val;

//                 // Mise à jour des running stats (EMA)
//                 running_mean_[c] = (1.0f - momentum_) * running_mean_[c]
//                                  +          momentum_  * mu;
//                 running_var_[c]  = (1.0f - momentum_) * running_var_[c]
//                                  +          momentum_  * var;
//             }

//             const float mu_used  = training_ ? mu : running_mean_[c];
//             const float inv_used = training_ ? inv_std_val 
//                                             : 1.0f / std::sqrt(running_var_[c] + eps_);

//             // --- Normalisation + scale/shift ---
//             int idx = 0;
//             for (int b = 0; b < B; ++b) {
//                 for (int d = 0; d < D; ++d) {
//                     for (int h = 0; h < H; ++h) {
//                         for (int w = 0; w < W; ++w) {
//                             const float x_val = input(b, c, d, h, w);
//                             const float xh = (x_val - mu_used) * inv_used;
//                             x_hat_[idx++] = xh;
//                             output(b, c, d, h, w) = gamma_[c] * xh + beta_[c];
//                         }
//                     }
//                 }
//             }
//         }
        
//         return output;
//     }

//     // -----------------------------------------------------------------------
//     //  Backward pass
//     //  gradOutput : gradient entrant (même shape que output)
//     //  Retourne le gradient par rapport à l'entrée
//     // -----------------------------------------------------------------------
//     Tensor backward(const Tensor& gradOutput) override {
//         const int B = shape_[0];
//         const int C = shape_[1];
//         const int D = shape_[2];
//         const int H = shape_[3];
//         const int W = shape_[4];
//         const int spatial = D * H * W;
//         const int M = B * spatial;

//         // Réinitialiser les gradients
//         std::fill(d_gamma_.begin(), d_gamma_.end(), 0.0f);
//         std::fill(d_beta_.begin(),  d_beta_.end(),  0.0f);

//         Tensor d_input(B, C, D, H, W);
//         d_input.setZero();

//         for (int c = 0; c < C; ++c) {
//             float dg = 0.0f, db = 0.0f;
            
//             // Accumulation des gradients gamma et beta
//             int idx = 0;
//             for (int b = 0; b < B; ++b) {
//                 for (int d = 0; d < D; ++d) {
//                     for (int h = 0; h < H; ++h) {
//                         for (int w = 0; w < W; ++w) {
//                             const float grad_out = gradOutput(b, c, d, h, w);
//                             dg += grad_out * x_hat_[idx];
//                             db += grad_out;
//                             idx++;
//                         }
//                     }
//                 }
//             }
//             d_gamma_[c] = dg;
//             d_beta_[c]  = db;

//             // Gradient par rapport à x_hat
//             // d_x = (1/M) * inv_std * [ M*d_xhat - sum(d_xhat)
//             //                           - x_hat * sum(d_xhat * x_hat) ]
//             float sum_dxhat = 0.0f;
//             float sum_dxhat_xhat = 0.0f;
            
//             idx = 0;
//             for (int b = 0; b < B; ++b) {
//                 for (int d = 0; d < D; ++d) {
//                     for (int h = 0; h < H; ++h) {
//                         for (int w = 0; w < W; ++w) {
//                             const float dxh = gradOutput(b, c, d, h, w) * gamma_[c];
//                             sum_dxhat += dxh;
//                             sum_dxhat_xhat += dxh * x_hat_[idx];
//                             idx++;
//                         }
//                     }
//                 }
//             }

//             const float inv_M = 1.0f / static_cast<float>(M);
//             idx = 0;
//             for (int b = 0; b < B; ++b) {
//                 for (int d = 0; d < D; ++d) {
//                     for (int h = 0; h < H; ++h) {
//                         for (int w = 0; w < W; ++w) {
//                             const float dxh = gradOutput(b, c, d, h, w) * gamma_[c];
//                             d_input(b, c, d, h, w) = inv_std_[c] * inv_M *
//                                 (static_cast<float>(M) * dxh - sum_dxhat - x_hat_[idx] * sum_dxhat_xhat);
//                             idx++;
//                         }
//                     }
//                 }
//             }
//         }
        
//         return d_input;
//     }

//     // -----------------------------------------------------------------------
//     //  Mise à jour des paramètres (appelée par l'optimiseur)
//     //  Cette méthode correspond à la signature de Layer::updateParams
//     // -----------------------------------------------------------------------
//     void updateParams(Optimizer& optimizer) override {
//         // Pour BatchNorm, nous devons mettre à jour gamma et beta
//         // Note: L'optimiseur doit gérer les paramètres de la couche
//         // Cette implémentation suppose que l'optimiseur a accès aux paramètres
        
//         // Alternative: si l'optimiseur ne gère pas directement les paramètres,
//         // nous pouvons faire la mise à jour manuellement
//         for (int c = 0; c < num_features_; ++c) {
//             // Mise à jour simple SGD (à remplacer par l'optimiseur si disponible)
//             gamma_[c] -= 0.001f * d_gamma_[c];  // learning rate par défaut
//             beta_[c]  -= 0.001f * d_beta_[c];
//         }
        
//         // Réinitialiser les gradients après mise à jour
//         std::fill(d_gamma_.begin(), d_gamma_.end(), 0.0f);
//         std::fill(d_beta_.begin(),  d_beta_.end(),  0.0f);
//     }

//     // -----------------------------------------------------------------------
//     //  Méthodes additionnelles pour BatchNorm
//     // -----------------------------------------------------------------------
    
//     void setTraining(bool training) { training_ = training; }
//     bool isTraining() const { return training_; }

//     // Accès direct aux paramètres (pour l'optimiseur)
//     std::vector<float>& getGamma() { return gamma_; }
//     std::vector<float>& getBeta() { return beta_; }
//     std::vector<float>& getGammaGrad() { return d_gamma_; }
//     std::vector<float>& getBetaGrad() { return d_beta_; }
    
//     const std::vector<float>& getGamma() const { return gamma_; }
//     const std::vector<float>& getBeta() const { return beta_; }
    
//     // Accès aux running stats
//     const std::vector<float>& runningMean() const { return running_mean_; }
//     const std::vector<float>& runningVar()  const { return running_var_; }
    
//     void setRunningMean(const std::vector<float>& mean) { running_mean_ = mean; }
//     void setRunningVar(const std::vector<float>& var)   { running_var_  = var; }

// private:
//     // -----------------------------------------------------------------------
//     //  Membres
//     // -----------------------------------------------------------------------
//     int   num_features_;
//     float eps_;
//     float momentum_;
//     bool  training_;

//     std::vector<float> gamma_;        // paramètre scale  (C)
//     std::vector<float> beta_;         // paramètre shift  (C)
//     std::vector<float> d_gamma_;      // gradient gamma
//     std::vector<float> d_beta_;       // gradient beta

//     std::vector<float> running_mean_; // EMA moyenne
//     std::vector<float> running_var_;  // EMA variance

//     // Buffers forward/backward
//     Tensor input_;                    // Sauvegarde de l'entrée
//     std::vector<int> shape_;          // Sauvegarde de la shape
//     std::vector<float> x_hat_;        // valeurs normalisées (avant gamma/beta)
//     std::vector<float> mean_;         // mu par canal (batch courant)
//     std::vector<float> inv_std_;      // 1/sqrt(var+eps) par canal
// };



#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <Eigen/Dense>
#include "Layer.hpp"
#include "Tensor.hpp"
#include "Optimizer.hpp"
#include "ModelSerializer.hpp"

// ---------------------------------------------------------------------------
//  BatchNorm3D — Batch Normalisation pour volumes 3D
//
//  Entrée  : Tensor (N, C, D, H, W)
//  Sortie  : Tensor (N, C, D, H, W)   — même shape
//
//  Pour chaque canal c :
//      mu_c    = mean  sur (N, D, H, W)
//      var_c   = var   sur (N, D, H, W)
//      x_hat   = (x - mu_c) / sqrt(var_c + eps)
//      y       = gamma_c * x_hat + beta_c
//
//  En inférence on utilise les running stats (EMA).
//
//  Paramètres apprenables mis à jour via l'optimiseur externe (Adam/SGD),
//  gamma_ et beta_ sont des Eigen::VectorXf pour compatibilité avec
//  Optimizer::updateBias().
// ---------------------------------------------------------------------------
class BatchNorm3D : public Layer {
public:
    // -----------------------------------------------------------------------
    //  Constructeur
    //  num_features : nombre de canaux C
    //  eps          : stabilité numérique (défaut 1e-5)
    //  momentum     : coefficient EMA pour les running stats (défaut 0.1)
    // -----------------------------------------------------------------------
    explicit BatchNorm3D(int   num_features,
                         float eps      = 1e-5f,
                         float momentum = 0.1f)
        : num_features_(num_features),
          eps_(eps),
          momentum_(momentum),
          training_(true)
    {
        // Paramètres apprenables (Eigen — compatibles Optimizer::updateBias)
        gamma_   = Eigen::VectorXf::Ones(num_features);
        beta_    = Eigen::VectorXf::Zero(num_features);
        d_gamma_ = Eigen::VectorXf::Zero(num_features);
        d_beta_  = Eigen::VectorXf::Zero(num_features);

        // Running statistics (inférence) — également Eigen pour cohérence
        running_mean_ = Eigen::VectorXf::Zero(num_features);
        running_var_  = Eigen::VectorXf::Ones(num_features);

        // Buffers scalaires par canal
        mean_.resize(num_features, 0.0f);
        inv_std_.resize(num_features, 1.0f);

        isTrainable = true;
    }

    // -----------------------------------------------------------------------
    //  Interface Layer
    // -----------------------------------------------------------------------

    std::string getName() const override { return "BatchNorm3D"; }

    void saveParameters(boost::archive::binary_oarchive& archive) const override {
        archive << gamma_;
        archive << beta_;
        archive << running_mean_;
        archive << running_var_;
    }

    void loadParameters(boost::archive::binary_iarchive& archive) override {
        archive >> gamma_;
        archive >> beta_;
        archive >> running_mean_;
        archive >> running_var_;
    }

    // -----------------------------------------------------------------------
    //  Forward pass
    // -----------------------------------------------------------------------
    Tensor forward(const Tensor& input) override {
        if (input.ndim() != 5)
            throw std::invalid_argument(
                "BatchNorm3D: expected 5D tensor (B,C,D,H,W), got "
                + std::to_string(input.ndim()) + "D");

        const int B = input.dim(0);
        const int C = input.dim(1);
        const int D = input.dim(2);
        const int H = input.dim(3);
        const int W = input.dim(4);

        if (C != num_features_)
            throw std::invalid_argument(
                "BatchNorm3D: expected " + std::to_string(num_features_)
                + " channels, got " + std::to_string(C));

        const int spatial = D * H * W;
        const int M       = B * spatial;   // nb de valeurs moyennées par canal

        // Sauvegarde pour le backward
        shape_ = input.shape();

        Tensor output(B, C, D, H, W);
        x_hat_.resize(B * C * D * H * W);

        for (int c = 0; c < C; ++c) {

            // ── Moyenne ──────────────────────────────────────────────────────
            float mu = 0.0f;
            for (int b = 0; b < B; ++b)
            for (int d = 0; d < D; ++d)
            for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                mu += input(b, c, d, h, w);
            mu /= static_cast<float>(M);

            // ── Variance ─────────────────────────────────────────────────────
            float var = 0.0f;
            for (int b = 0; b < B; ++b)
            for (int d = 0; d < D; ++d)
            for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                float diff = input(b, c, d, h, w) - mu;
                var += diff * diff;
            }
            var /= static_cast<float>(M);

            const float inv_std_val = 1.0f / std::sqrt(var + eps_);

            if (training_) {
                mean_[c]    = mu;
                inv_std_[c] = inv_std_val;
                // EMA running stats
                running_mean_[c] = (1.0f - momentum_) * running_mean_[c]
                                 +          momentum_  * mu;
                running_var_[c]  = (1.0f - momentum_) * running_var_[c]
                                 +          momentum_  * var;
            }

            const float mu_used  = training_ ? mu          : running_mean_[c];
            const float inv_used = training_ ? inv_std_val
                                             : 1.0f / std::sqrt(running_var_[c] + eps_);

            // ── Normalisation + scale/shift ───────────────────────────────────
            // x_hat_ indexé en (b, c, d, h, w) avec c en dimension majeure
            // après b : offset = b*C*spatial + c*spatial + position_spatiale
            for (int b = 0; b < B; ++b)
            for (int d = 0; d < D; ++d)
            for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                const int idx = ((b * C + c) * D + d) * H * W + h * W + w;
                const float xh = (input(b, c, d, h, w) - mu_used) * inv_used;
                x_hat_[idx] = xh;
                output(b, c, d, h, w) = gamma_[c] * xh + beta_[c];
            }
        }

        return output;
    }

    // -----------------------------------------------------------------------
    //  Backward pass
    //  Formule standard BatchNorm :
    //    d_x = (1/M) * inv_std * [ M*d_xhat - sum(d_xhat)
    //                              - x_hat * sum(d_xhat * x_hat) ]
    //  avec d_xhat = grad_out * gamma
    // -----------------------------------------------------------------------
    Tensor backward(const Tensor& gradOutput) override {
        const int B = shape_[0];
        const int C = shape_[1];
        const int D = shape_[2];
        const int H = shape_[3];
        const int W = shape_[4];
        const int M = B * D * H * W;

        // Réinitialiser les gradients des paramètres
        d_gamma_.setZero();
        d_beta_.setZero();

        Tensor d_input(B, C, D, H, W);
        d_input.setZero();

        for (int c = 0; c < C; ++c) {

            // ── Gradients gamma et beta ───────────────────────────────────────
            float dg = 0.0f, db = 0.0f;
            for (int b = 0; b < B; ++b)
            for (int d = 0; d < D; ++d)
            for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                const int   idx      = ((b * C + c) * D + d) * H * W + h * W + w;
                const float grad_out = gradOutput(b, c, d, h, w);
                dg += grad_out * x_hat_[idx];
                db += grad_out;
            }
            d_gamma_[c] = dg;
            d_beta_[c]  = db;

            // ── Sommes pour le gradient d'entrée ─────────────────────────────
            float sum_dxhat      = 0.0f;
            float sum_dxhat_xhat = 0.0f;
            for (int b = 0; b < B; ++b)
            for (int d = 0; d < D; ++d)
            for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                const int   idx = ((b * C + c) * D + d) * H * W + h * W + w;
                const float dxh = gradOutput(b, c, d, h, w) * gamma_[c];
                sum_dxhat      += dxh;
                sum_dxhat_xhat += dxh * x_hat_[idx];
            }

            // ── Gradient d'entrée ─────────────────────────────────────────────
            const float inv_M = 1.0f / static_cast<float>(M);
            for (int b = 0; b < B; ++b)
            for (int d = 0; d < D; ++d)
            for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                const int   idx = ((b * C + c) * D + d) * H * W + h * W + w;
                const float dxh = gradOutput(b, c, d, h, w) * gamma_[c];
                d_input(b, c, d, h, w) =
                    inv_std_[c] * inv_M *
                    (static_cast<float>(M) * dxh
                     - sum_dxhat
                     - x_hat_[idx] * sum_dxhat_xhat);
            }
        }

        return d_input;
    }

    // -----------------------------------------------------------------------
    //  Mise à jour des paramètres via l'optimiseur externe (Adam/SGD)
    //  gamma_ et beta_ sont des Eigen::VectorXf — compatibles updateBias()
    // -----------------------------------------------------------------------
    void updateParams(Optimizer& optimizer) override {
        optimizer.updateBias(gamma_, d_gamma_);
        optimizer.updateBias(beta_,  d_beta_);
        d_gamma_.setZero();
        d_beta_.setZero();
    }

    // -----------------------------------------------------------------------
    //  Accesseurs
    // -----------------------------------------------------------------------
    void setTraining(bool training) { training_ = training; }
    bool isTraining() const         { return training_; }

    Eigen::VectorXf&       getGamma()     { return gamma_;   }
    Eigen::VectorXf&       getBeta()      { return beta_;    }
    Eigen::VectorXf&       getGammaGrad() { return d_gamma_; }
    Eigen::VectorXf&       getBetaGrad()  { return d_beta_;  }

    int numParams() const override { return gamma_.size() + beta_.size(); }

    const Eigen::VectorXf& getGamma()     const { return gamma_;   }
    const Eigen::VectorXf& getBeta()      const { return beta_;    }

    const Eigen::VectorXf& runningMean() const { return running_mean_; }
    const Eigen::VectorXf& runningVar()  const { return running_var_;  }
    void setRunningMean(const Eigen::VectorXf& mean) { running_mean_ = mean; }
    void setRunningVar (const Eigen::VectorXf& var)  { running_var_  = var;  }

private:
    int   num_features_;
    float eps_;
    float momentum_;
    bool  training_;

    // Paramètres apprenables — Eigen pour compatibilité Optimizer::updateBias
    Eigen::VectorXf gamma_, beta_;
    Eigen::VectorXf d_gamma_, d_beta_;

    // Running statistics
    Eigen::VectorXf running_mean_, running_var_;

    // Buffers forward/backward
    std::vector<int>   shape_;
    std::vector<float> x_hat_;     // valeurs normalisées — indexées (b,c,d,h,w)
    std::vector<float> mean_;      // mu par canal (batch courant)
    std::vector<float> inv_std_;   // 1/sqrt(var+eps) par canal
};