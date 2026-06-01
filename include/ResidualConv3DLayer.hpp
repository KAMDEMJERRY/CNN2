#pragma once
#include "Layer.hpp"
#include "ConvLayer3D.hpp"
#include "BatchNorm3DLayer.hpp"
#include "ActivationLayer.hpp"
#include <memory>
#include <string>

// =============================================================================
// ResidualConv3DLayer — Bloc résiduel 3D (style ResNet)
// =============================================================================
//
// Architecture interne :
//
//   input ──────────────────────────────────┐ (skip)
//     │                                     │
//     ▼                                     │
//   Conv3D(3³, s) → BN → ReLU             │
//     │                                     │
//     ▼                                     │
//   Conv3D(3³, 1) → BN                    │
//     │                                     │
//     └──────────────(+)───────────────────┘
//                     │
//                   ReLU
//                     │
//                  output
//
// Connexion résiduelle :
//   - Identité si (in_channels == out_channels && stride == 1)
//   - Projection 1×1×1 sinon (conv de rééchantillonnage)
//
// Usage :
//   model.addLayer(std::make_shared<ResidualConv3DLayer>(16, 32, 2));
//   // → bloc résiduel 16→32 canaux, stride 2
//
// Paramètres :
//   in_channels  : canaux d'entrée
//   out_channels : canaux de sortie
//   stride       : stride de la première conv (réduit la résolution)
//
// =============================================================================

class ResidualConv3DLayer : public Layer {
public:

    // ── Constructeur ──────────────────────────────────────────────────────────
    ResidualConv3DLayer(int in_channels, int out_channels, int stride = 1)
        : in_ch_(in_channels), out_ch_(out_channels), stride_(stride)
    {
        if (in_channels <= 0 || out_channels <= 0 || stride <= 0)
            throw std::invalid_argument("[ResidualConv3D] Dimensions invalides");

        // ── Chemin principal ──────────────────────────────────────────────────
        // Conv1 : réduit spatialement si stride > 1
        conv1_ = std::make_shared<ConvLayer3D>(
            in_channels, out_channels,
            3, 3, 3,        // kernel 3×3×3
            stride, stride, stride,
            1, 1, 1);       // padding 1→ preserve (D/stride, H/stride, W/stride)

        bn1_   = std::make_shared<BatchNorm3D>(out_channels);
        relu1_ = std::make_shared<ReLULayer>();

        // Conv2 : affine, ne change pas les dimensions
        conv2_ = std::make_shared<ConvLayer3D>(
            out_channels, out_channels,
            3, 3, 3,
            1, 1, 1,        // stride 1
            1, 1, 1);

        bn2_ = std::make_shared<BatchNorm3D>(out_channels);

        // ── Connexion résiduelle ──────────────────────────────────────────────
        needs_proj_ = (in_channels != out_channels || stride != 1);
        if (needs_proj_) {
            // Projection 1×1×1 pour aligner canaux + stride
            proj_ = std::make_shared<ConvLayer3D>(
                in_channels, out_channels,
                1, 1, 1,
                stride, stride, stride,
                0, 0, 0);
            proj_bn_ = std::make_shared<BatchNorm3D>(out_channels);
        }

        relu_out_ = std::make_shared<ReLULayer>();

        isTrainable = true;
    }

    ~ResidualConv3DLayer() override = default;

    ResidualConv3DLayer(const ResidualConv3DLayer&)            = delete;
    ResidualConv3DLayer& operator=(const ResidualConv3DLayer&) = delete;

    // =========================================================================
    // FORWARD
    // =========================================================================
    Tensor forward(const Tensor& input) override {
        if (input.ndim() != 5)
            throw std::runtime_error(
                "[ResidualConv3D::forward] Attend un Tensor 5D (B,C,D,H,W)");
        if (input.dim(1) != in_ch_)
            throw std::runtime_error(
                "[ResidualConv3D::forward] Mauvais nombre de canaux d'entrée : "
                + std::to_string(input.dim(1)) + " vs " + std::to_string(in_ch_));

        input_cache_ = input;

        // ── Chemin principal : Conv1 → BN → ReLU → Conv2 → BN ────────────────
        Tensor x = conv1_ ->forward(input);
        x        = bn1_   ->forward(x);
        x        = relu1_ ->forward(x);
        x        = conv2_ ->forward(x);
        x        = bn2_   ->forward(x);

        // ── Connexion résiduelle ──────────────────────────────────────────────
        Tensor skip = needs_proj_
            ? proj_bn_->forward(proj_->forward(input))
            : input;

        // Vérification des dimensions (doit être identique)
        if (x.size() != skip.size())
            throw std::runtime_error(
                "[ResidualConv3D::forward] Dimensions main/skip incompatibles");

        for (int i = 0; i < x.size(); ++i)
            x[i] += skip[i];

        return relu_out_->forward(x);
    }

    // =========================================================================
    // BACKWARD
    // =========================================================================
    Tensor backward(const Tensor& grad_output) override {
        // ── ReLU de sortie ────────────────────────────────────────────────────
        Tensor grad = relu_out_->backward(grad_output);

        // Le gradient de la somme se propage aux deux chemins
        Tensor grad_skip = grad;   // copie pour le chemin skip

        // ── Chemin principal : BN2 → Conv2 → ReLU → BN1 → Conv1 ──────────────
        grad = bn2_   ->backward(grad);
        grad = conv2_ ->backward(grad);
        grad = relu1_ ->backward(grad);
        grad = bn1_   ->backward(grad);
        grad = conv1_ ->backward(grad);

        // ── Chemin skip ───────────────────────────────────────────────────────
        if (needs_proj_) {
            Tensor grad_proj = proj_bn_->backward(grad_skip);
            grad_proj        = proj_   ->backward(grad_proj);
            for (int i = 0; i < grad.size(); ++i)
                grad[i] += grad_proj[i];
        } else {
            for (int i = 0; i < grad.size(); ++i)
                grad[i] += grad_skip[i];
        }

        return grad;
    }

    // ── Mise à jour des poids ─────────────────────────────────────────────────
    void updateParams(Optimizer& optimizer) override {
        conv1_ ->updateParams(optimizer);
        bn1_   ->updateParams(optimizer);
        conv2_ ->updateParams(optimizer);
        bn2_   ->updateParams(optimizer);
        if (needs_proj_) {
            proj_   ->updateParams(optimizer);
            proj_bn_->updateParams(optimizer);
        }
    }

    // ── Sérialisation ─────────────────────────────────────────────────────────
    void saveParameters(boost::archive::binary_oarchive& ar) const override {
        conv1_ ->saveParameters(ar);
        bn1_   ->saveParameters(ar);
        conv2_ ->saveParameters(ar);
        bn2_   ->saveParameters(ar);
        if (needs_proj_) {
            proj_   ->saveParameters(ar);
            proj_bn_->saveParameters(ar);
        }
    }

    void loadParameters(boost::archive::binary_iarchive& ar) override {
        conv1_ ->loadParameters(ar);
        bn1_   ->loadParameters(ar);
        conv2_ ->loadParameters(ar);
        bn2_   ->loadParameters(ar);
        if (needs_proj_) {
            proj_   ->loadParameters(ar);
            proj_bn_->loadParameters(ar);
        }
    }

    // ── Informations ─────────────────────────────────────────────────────────
    std::string getName() const override {
        return "ResidualConv3D(in=" + std::to_string(in_ch_)
             + " out="  + std::to_string(out_ch_)
             + " s="    + std::to_string(stride_)
             + (needs_proj_ ? " proj" : "") + ")";
    }

    int numParams() const override {
        int p = conv1_->numParams() + bn1_->numParams()
              + conv2_->numParams() + bn2_->numParams();
        if (needs_proj_) p += proj_->numParams() + proj_bn_->numParams();
        return p;
    }

    // ── Accesseurs (tests) ────────────────────────────────────────────────────
    ConvLayer3D& getConv1() { return *conv1_; }
    ConvLayer3D& getConv2() { return *conv2_; }
    bool hasProjection()    const { return needs_proj_; }

private:
    int  in_ch_, out_ch_, stride_;
    bool needs_proj_ = false;

    // Couches internes
    std::shared_ptr<ConvLayer3D>  conv1_, conv2_;
    std::shared_ptr<BatchNorm3D>  bn1_,  bn2_;
    std::shared_ptr<ReLULayer>    relu1_, relu_out_;

    // Projection (si nécessaire)
    std::shared_ptr<ConvLayer3D> proj_;
    std::shared_ptr<BatchNorm3D> proj_bn_;

    // Cache backward
    Tensor input_cache_;
};
