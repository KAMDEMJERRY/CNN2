#pragma once
#include "Layer.hpp"
#include "SparseTensor.hpp"
#include "SparseConvLayer3D.hpp"

// =============================================================================
// SparseConvAdapterLayer
// =============================================================================
// Adaptateur qui fait hériter SparseConvLayer3D de l'interface Layer.
//
// Principe :
//   forward  : Tensor dense → SparseTensor → SparseConvLayer3D → Tensor dense
//   backward : Tensor grad  → SparseTensor grad → backward sparse → Tensor grad
//
// Cela permet d'utiliser une convolution sparse exactement comme une couche
// dense dans CNN::addLayer(), sans modifier Layer, CNN, ni aucune autre classe.
//
//   model.addLayer(std::make_shared<SparseConvAdapterLayer>(
//       1, 16, 3,3,3, 1,1,1, 1,1,1, true, 0.02f));   // SubManifold
//   model.addLayer(std::make_shared<ReLULayer>());
//   model.addLayer(std::make_shared<SparseConvAdapterLayer>(
//       16, 32, 3,3,3, 2,2,2, 1,1,1, false, 0.0f));   // Standard stride=2
//
// Conversions par couche :
//   from_dense() en entrée  : O(B·D·H·W)  — seuillage
//   to_dense()   en sortie  : O(nnz · C)  — reconstruction
//   Ces surcoûts sont compensés par le gain de l'im2col sparse.
//
// ReLU sparse :
//   SparseReLUAdapterLayer est fourni séparément. Contrairement à ReLULayer
//   (qui opère sur Tensor dense), elle applique le ReLU uniquement sur les
//   features actives du SparseTensor, puis retourne un Tensor dense.
//   Cela évite une conversion inutile entre SparseConvAdapterLayer et la
//   couche ReLU suivante si tu chaînes plusieurs blocs sparse.
//
// SparseGlobalAvgPoolLayer :
//   Remplace GlobalAvgPool3DLayer en fin de bloc sparse.
//   Calcule la moyenne uniquement sur les voxels actifs (plus juste pour
//   les volumes creux que de diviser par D*H*W).
// =============================================================================


// =============================================================================
// SparseConvAdapterLayer — couche principale
// =============================================================================
class SparseConvAdapterLayer : public Layer {
public:

    // ── Constructeur ──────────────────────────────────────────────────────────
    //
    // Paramètres identiques à SparseConvLayer3D + threshold pour from_dense().
    //
    // submanifold = true  → SubManifold (stride=1 obligatoire, sparsité préservée)
    // submanifold = false → Standard sparse (stride>1 possible, sparsité élargie)
    // threshold           → seuil d'activation pour from_dense() en entrée
    //                       0.0f = conserve tout sauf strictement nul
    //                       1e-4f recommandé pour CT-scan normalisé [0,1]
    SparseConvAdapterLayer(int in_channels,  int out_channels,
                           int kernel_d,     int kernel_h,   int kernel_w,
                           int stride_d = 1, int stride_h = 1, int stride_w = 1,
                           int pad_d    = 0, int pad_h    = 0, int pad_w    = 0,
                           bool  submanifold = true,
                           float threshold   = 0.02f)
        : sparse_conv_(in_channels, out_channels,
                       kernel_d, kernel_h, kernel_w,
                       stride_d, stride_h, stride_w,
                       pad_d,    pad_h,    pad_w,
                       submanifold)
        , threshold_(threshold)
    {
        isTrainable = true;
    }

    ~SparseConvAdapterLayer() override = default;

    // Non copiable (cache interne)
    SparseConvAdapterLayer(const SparseConvAdapterLayer&)            = delete;
    SparseConvAdapterLayer& operator=(const SparseConvAdapterLayer&) = delete;

    // ── Forward ───────────────────────────────────────────────────────────────
    //
    // 1. Tensor dense → SparseTensor  (seuillage par threshold_)
    // 2. Convolution sparse            (im2col sparse + GEMM)
    // 3. SparseTensor → Tensor dense  (reconstruction)
    //
    // Le SparseTensor de sortie est mis en cache pour le backward du ReLU
    // suivant et pour le backward de cette couche.
    Tensor forward(const Tensor& input) override {

        if (input.ndim() != 5)
            throw std::runtime_error(
                "[SparseConvAdapterLayer] Attend un Tensor 5D (B,C,D,H,W). "
                "Reçu ndim=" + std::to_string(input.ndim()));

        // ── 1. Dense → Sparse ─────────────────────────────────────────────────
        sp_input_cache_ = SparseTensor::from_dense(input, threshold_);

        // ── 2. Convolution sparse ─────────────────────────────────────────────
        sp_output_cache_ = sparse_conv_.forward(sp_input_cache_);

        // ── 3. Sparse → Dense ─────────────────────────────────────────────────
        return sp_output_cache_.to_dense();
    }

    // ── Backward ──────────────────────────────────────────────────────────────
    //
    // Reçoit le gradient dense dL/dY (B, C_out, D_out, H_out, W_out).
    //
    // 1. Tensor grad dense → SparseTensor grad
    //    On utilise les coordonnées du cache de sortie (sp_output_cache_) :
    //    seuls les voxels qui étaient actifs en sortie reçoivent un gradient.
    //    Les voxels inactifs ont une contribution nulle → on les ignore.
    //
    // 2. Backward sparse → SparseTensor grad d'entrée
    //
    // 3. SparseTensor grad d'entrée → Tensor dense
    //    Les positions inactives sont mises à 0.
    Tensor backward(const Tensor& grad_output) override {

        // ── 1. Grad dense → SparseTensor grad ─────────────────────────────────
        // On réutilise les coords de la sortie du forward pour ne garder
        // que les gradients aux positions actives.
        const int N_out = sp_output_cache_.nnz();
        const int C_out = sp_output_cache_.num_channels;

        SparseTensor sp_grad_out;
        sp_grad_out.batch_size   = sp_output_cache_.batch_size;
        sp_grad_out.num_channels = C_out;
        sp_grad_out.spatial_d    = sp_output_cache_.spatial_d;
        sp_grad_out.spatial_h    = sp_output_cache_.spatial_h;
        sp_grad_out.spatial_w    = sp_output_cache_.spatial_w;
        sp_grad_out.coords       = sp_output_cache_.coords;  // même positions
        sp_grad_out.features.resize(N_out, C_out);

        // Collecte les gradients aux positions actives
        for (int i = 0; i < N_out; ++i) {
            const int b = sp_output_cache_.coords(i, 0);
            const int d = sp_output_cache_.coords(i, 1);
            const int h = sp_output_cache_.coords(i, 2);
            const int w = sp_output_cache_.coords(i, 3);
            for (int c = 0; c < C_out; ++c)
                sp_grad_out.features(i, c) = grad_output(b, c, d, h, w);
        }

        // ── 2. Backward sparse ────────────────────────────────────────────────
        SparseTensor sp_grad_in = sparse_conv_.backward(sp_grad_out);

        // ── 3. SparseTensor grad → Tensor dense ───────────────────────────────
        return sp_grad_in.to_dense();
    }

    // ── Mise à jour des poids ─────────────────────────────────────────────────
    void updateParams(Optimizer& optimizer) override {
        sparse_conv_.updateParams(optimizer);
    }

    std::string getName() const override {
        return "SparseConvAdapter("
             + std::string(sparse_conv_.isSubmanifold() ? "SubManifold" : "Standard")
             + ")";
    }

    // ── Accesseurs utiles ─────────────────────────────────────────────────────

    // Densité du SparseTensor d'entrée du dernier forward (pour monitoring)
    float lastInputDensity()  const { return sp_input_cache_.density();  }
    float lastOutputDensity() const { return sp_output_cache_.density(); }
    int   lastInputNnz()      const { return sp_input_cache_.nnz();      }
    int   lastOutputNnz()     const { return sp_output_cache_.nnz();     }

    SparseConvLayer3D& getSparseConv() { return sparse_conv_; }

private:

    SparseConvLayer3D sparse_conv_;
    float             threshold_;

    // Cache des SparseTensors d'entrée et sortie du dernier forward.
    // Nécessaires pour :
    //   - sp_input_cache_  : backward de sparse_conv_
    //   - sp_output_cache_ : reconstruction du grad sparse en backward
    SparseTensor sp_input_cache_;
    SparseTensor sp_output_cache_;
};


// =============================================================================
// SparseReLUAdapterLayer
// =============================================================================
// ReLU opérant directement sur les features du SparseTensor interne.
//
// Différence avec ReLULayer :
//   ReLULayer     : opère sur un Tensor dense (tous les voxels, y compris nuls)
//   SparseReLU    : opère uniquement sur les nnz features actives
//
// Usage : remplace ReLULayer après SparseConvAdapterLayer pour éviter une
// conversion dense inutile. Résultat identique mathématiquement.
//
// Note : cette couche reçoit le Tensor dense produit par to_dense() de la
// couche précédente. Elle le reconvertit en sparse, applique ReLU, reconvertit.
// Si tu veux éviter totalement les conversions intermédiaires, utilise
// SparseConvAdapterLayer::forward() avec ReLU intégré (via applyReLU),
// puis appelle to_dense() une seule fois en fin de bloc sparse.
// =============================================================================
class SparseReLUAdapterLayer : public Layer {
public:

    explicit SparseReLUAdapterLayer(float threshold = 0.0f)
        : threshold_(threshold) {}

    ~SparseReLUAdapterLayer() override = default;

    // Forward : Tensor → sparse features → ReLU → Tensor
    Tensor forward(const Tensor& input) override {

        // Cache pour le backward (masque de signe)
        input_cache_ = input;

        // Conversion, ReLU in-place sur les features, reconversion
        SparseTensor sp = SparseTensor::from_dense(input, threshold_);
        sp.applyReLU();
        return sp.to_dense();
    }

    // Backward : masque binaire identique à ReLULayer
    // input_cache_ > 0 → gradient passe, sinon bloqué
    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_input(grad_output.shape());
        for (int i = 0; i < grad_output.size(); ++i)
            grad_input[i] = (input_cache_[i] > 0.f) ? grad_output[i] : 0.f;
        return grad_input;
    }

    std::string getName() const override { return "SparseReLUAdapter"; }

private:
    float  threshold_;
    Tensor input_cache_;
};


// =============================================================================
// SparseGlobalAvgPoolLayer
// =============================================================================
// Remplace GlobalAvgPool3DLayer à la fin du bloc sparse.
//
// Différence clé avec GlobalAvgPool3DLayer :
//   Dense  : moyenne = sum / (D * H * W)       — divise par tous les voxels
//   Sparse : moyenne = sum / nnz_par_batch      — divise par les voxels actifs
//
// Sur un volume médical creux (os ≈ 15-20% du volume), diviser par D*H*W
// sous-estime les activations d'un facteur ~5-7x. La version sparse est
// donc plus juste physiquement : elle représente la feature moyenne sur
// la matière active, pas sur le volume total incluant le fond.
//
// Sortie : Tensor dense (B, C, 1, 1, 1) compatible DenseLayer, identique
//          à GlobalAvgPool3DLayer.
// =============================================================================
class SparseGlobalAvgPoolLayer : public Layer {
public:

    explicit SparseGlobalAvgPoolLayer(float threshold = 0.0f)
        : threshold_(threshold) {}

    ~SparseGlobalAvgPoolLayer() override = default;

    // Forward : Tensor (B,C,D,H,W) → sparse globalAvgPool → Tensor (B,C,1,1,1)
    Tensor forward(const Tensor& input) override {

        if (input.ndim() != 5)
            throw std::runtime_error(
                "[SparseGlobalAvgPoolLayer] Attend un Tensor 5D (B,C,D,H,W)");

        SparseTensor sp = SparseTensor::from_dense(input, threshold_);

        // Cache pour le backward
        cached_sp_    = sp;
        cached_shape_ = input.shape();

        return sp.globalAvgPool();
    }

    // Backward : redistribue grad_output uniformément sur les nnz voxels actifs
    // (même logique que GlobalAvgPool3DLayer mais normalisé par nnz, pas D*H*W)
    Tensor backward(const Tensor& grad_output) override {

        const int B   = grad_output.dim(0);
        const int C   = grad_output.dim(1);
        const int N   = cached_sp_.nnz();

        // Gradient d'entrée dense : zéro partout sauf aux positions actives
        Tensor grad_input(cached_shape_);
        grad_input.setZero();

        // Compteur de voxels actifs par batch
        Eigen::VectorXi counts = Eigen::VectorXi::Zero(B);
        for (int i = 0; i < N; ++i)
            ++counts[cached_sp_.coords(i, 0)];

        // Redistribution uniforme
        for (int i = 0; i < N; ++i) {
            const int b = cached_sp_.coords(i, 0);
            const int d = cached_sp_.coords(i, 1);
            const int h = cached_sp_.coords(i, 2);
            const int w = cached_sp_.coords(i, 3);
            const float denom = (counts[b] > 0)
                ? static_cast<float>(counts[b]) : 1.f;
            for (int c = 0; c < C; ++c)
                grad_input(b, c, d, h, w) =
                    grad_output(b, c, 0, 0, 0) / denom;
        }

        return grad_input;
    }

    std::string getName() const override { return "SparseGlobalAvgPool"; }

    // Densité du dernier forward (utile pour monitoring)
    float lastDensity() const { return cached_sp_.density(); }
    int   lastNnz()     const { return cached_sp_.nnz();     }

private:
    float            threshold_;
    SparseTensor     cached_sp_;
    std::vector<int> cached_shape_;
};