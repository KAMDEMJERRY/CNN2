// tests/test_ConvLayer.cpp

#include <gtest/gtest.h>
#include "ConvLayer.hpp"
#include "Optimizer.hpp"
#include <cmath>

static constexpr float EPS = 1e-4f;

// =============================================================================
// Helpers
// =============================================================================

// Crée un Tensor [B,C,H,W] rempli d'une constante
static Tensor makeConst(int B, int C, int H, int W, float val) {
    Tensor t(B, C, H, W);
    t.setConstant(val);
    return t;
}

// Formule standard de dimension de sortie convolutive
static int outDim(int in, int pad, int kernel, int stride) {
    return (in + 2 * pad - kernel) / stride + 1;
}

// =============================================================================
// ── Construction ─────────────────────────────────────────────────────────────
// =============================================================================

TEST(ConvLayerConstruction, IsTrainable) {
    ConvLayer conv(1, 4, 3, 3);
    EXPECT_TRUE(conv.isTrainable);
}

TEST(ConvLayerConstruction, GetName) {
    ConvLayer conv(1, 4, 3, 3);
    EXPECT_EQ(conv.getName(), "ConvLayer");
}

TEST(ConvLayerConstruction, WeightsShape) {
    // [out_ch, in_ch, kH, kW]
    ConvLayer conv(2, 8, 3, 5);
    const Tensor& w = conv.getWeights();
    EXPECT_EQ(w.dim(0), 8);
    EXPECT_EQ(w.dim(1), 2);
    EXPECT_EQ(w.dim(2), 3);
    EXPECT_EQ(w.dim(3), 5);
}

TEST(ConvLayerConstruction, BiasShape) {
    ConvLayer conv(1, 6, 3, 3);
    EXPECT_EQ(conv.getBias().size(), 6);
}

TEST(ConvLayerConstruction, BiasInitializedToZero) {
    ConvLayer conv(1, 4, 3, 3);
    for (int i = 0; i < conv.getBias().size(); ++i)
        EXPECT_FLOAT_EQ(conv.getBias()(i), 0.0f);
}

TEST(ConvLayerConstruction, WeightsNonZeroAfterHe) {
    ConvLayer conv(1, 4, 3, 3);
    conv.initializeWeights("he");
    float norm = 0.f;
    for (int i = 0; i < conv.getWeights().size(); ++i)
        norm += conv.getWeights()[i] * conv.getWeights()[i];
    EXPECT_GT(norm, 0.f);
}

TEST(ConvLayerConstruction, WeightsNonZeroAfterXavier) {
    ConvLayer conv(1, 4, 3, 3);
    conv.initializeWeights("xavier");
    float norm = 0.f;
    for (int i = 0; i < conv.getWeights().size(); ++i)
        norm += conv.getWeights()[i] * conv.getWeights()[i];
    EXPECT_GT(norm, 0.f);
}

TEST(ConvLayerConstruction, NotCopyable) {
    EXPECT_FALSE(std::is_copy_constructible_v<ConvLayer>);
    EXPECT_FALSE(std::is_copy_assignable_v<ConvLayer>);
}

// =============================================================================
// ── Dimensions de sortie (forward) ───────────────────────────────────────────
// =============================================================================

TEST(ConvLayerForward, OutputShape_NoPadding) {
    ConvLayer conv(1, 4, 3, 3, 1, 1, 0, 0);
    Tensor input(2, 1, 8, 8);
    input.setRandom();
    Tensor out = conv.forward(input);

    EXPECT_EQ(out.dim(0), 2);
    EXPECT_EQ(out.dim(1), 4);
    EXPECT_EQ(out.dim(2), outDim(8, 0, 3, 1));  // 6
    EXPECT_EQ(out.dim(3), outDim(8, 0, 3, 1));  // 6
}

TEST(ConvLayerForward, OutputShape_SamePadding) {
    // kernel=3, pad=1 → H et W conservés
    ConvLayer conv(1, 8, 3, 3, 1, 1, 1, 1);
    Tensor input(2, 1, 14, 14);
    input.setRandom();
    Tensor out = conv.forward(input);

    EXPECT_EQ(out.dim(2), 14);
    EXPECT_EQ(out.dim(3), 14);
}

TEST(ConvLayerForward, OutputShape_Stride2) {
    ConvLayer conv(1, 4, 3, 3, 2, 2, 0, 0);
    Tensor input(1, 1, 14, 14);
    input.setRandom();
    Tensor out = conv.forward(input);

    EXPECT_EQ(out.dim(2), outDim(14, 0, 3, 2));  // 6
    EXPECT_EQ(out.dim(3), outDim(14, 0, 3, 2));  // 6
}

TEST(ConvLayerForward, OutputShape_AsymmetricKernel) {
    // kernel 1×5, pad 0×2 → H: (8-1)/1+1=8, W: (8+4-5)/1+1=8
    ConvLayer conv(1, 4, 1, 5, 1, 1, 0, 2);
    Tensor input(1, 1, 8, 8);
    input.setRandom();
    Tensor out = conv.forward(input);

    EXPECT_EQ(out.dim(2), outDim(8, 0, 1, 1));  // 8
    EXPECT_EQ(out.dim(3), outDim(8, 2, 5, 1));  // 8
}

TEST(ConvLayerForward, OutputShape_BatchPreserved) {
    ConvLayer conv(1, 4, 3, 3, 1, 1, 1, 1);
    for (int B : {1, 4, 16}) {
        Tensor input(B, 1, 8, 8);
        input.setRandom();
        EXPECT_EQ(conv.forward(input).dim(0), B);
    }
}

TEST(ConvLayerForward, OutputShape_MultiChannel) {
    ConvLayer conv(3, 16, 3, 3, 1, 1, 1, 1);
    Tensor input(2, 3, 8, 8);
    input.setRandom();
    Tensor out = conv.forward(input);
    EXPECT_EQ(out.dim(1), 16);
}

// =============================================================================
// ── Valeurs de sortie (forward) ───────────────────────────────────────────────
// =============================================================================

TEST(ConvLayerForward, Kernel1x1_WeightOne_OutputEqualsInput) {
    ConvLayer conv(1, 1, 1, 1);
    Tensor w(1, 1, 1, 1);
    w.setConstant(1.0f);
    conv.setWeights(w);
    conv.setBias(Eigen::VectorXf::Zero(1));

    Tensor input = makeConst(1, 1, 4, 4, 5.0f);
    Tensor out   = conv.forward(input);

    for (int i = 0; i < out.size(); ++i)
        EXPECT_NEAR(out[i], 5.0f, EPS);
}

TEST(ConvLayerForward, Kernel1x1_WeightZero_OutputIsBiasOnly) {
    ConvLayer conv(1, 1, 1, 1);
    Tensor w(1, 1, 1, 1);
    w.setZero();
    conv.setWeights(w);
    Eigen::VectorXf b(1);
    b(0) = 3.0f;
    conv.setBias(b);

    Tensor input = makeConst(1, 1, 4, 4, 99.0f);
    Tensor out   = conv.forward(input);

    for (int i = 0; i < out.size(); ++i)
        EXPECT_NEAR(out[i], 3.0f, EPS);
}

TEST(ConvLayerForward, ZeroPaddingFilledWithZero) {
    // Avec un kernel qui déborde et weights=1, les pixels de padding
    // doivent contribuer 0 (zero-padding)
    ConvLayer conv(1, 1, 3, 3, 1, 1, 0, 0);
    Tensor w(1, 1, 3, 3);
    w.setConstant(1.0f);
    conv.setWeights(w);
    conv.setBias(Eigen::VectorXf::Zero(1));

    // Input 3×3 rempli de 1 — un seul patch valide (le centre)
    // somme = 9 voisins × 1 = 9
    Tensor input = makeConst(1, 1, 3, 3, 1.0f);
    Tensor out   = conv.forward(input);  // out: 1×1×1×1

    EXPECT_EQ(out.dim(2), 1);
    EXPECT_NEAR(out(0, 0, 0, 0), 9.0f, EPS);
}

TEST(ConvLayerForward, KnownConvolution_2x2Kernel) {
    // input 4×4, kernel 2×2, stride 1, pad 0 → out 3×3
    // weights = [[1,0],[0,1]] (identité 2×2) → out(i,j) = in(i,j) + in(i+1,j+1)
    ConvLayer conv(1, 1, 2, 2, 1, 1, 0, 0);
    Tensor w(1, 1, 2, 2);
    w.setZero();
    w(0, 0, 0, 0) = 1.0f;
    w(0, 0, 1, 1) = 1.0f;
    conv.setWeights(w);
    conv.setBias(Eigen::VectorXf::Zero(1));

    Tensor input(1, 1, 4, 4);
    input.setZero();
    // Rempli d'indices pour traçabilité
    for (int h = 0; h < 4; ++h)
        for (int ww = 0; ww < 4; ++ww)
            input(0, 0, h, ww) = static_cast<float>(h * 4 + ww);

    Tensor out = conv.forward(input);
    EXPECT_EQ(out.dim(2), 3);

    // out(0,0) = input(0,0) + input(1,1) = 0 + 5 = 5
    EXPECT_NEAR(out(0, 0, 0, 0), 5.0f, EPS);
    // out(0,1) = input(0,1) + input(1,2) = 1 + 6 = 7
    EXPECT_NEAR(out(0, 0, 0, 1), 7.0f, EPS);
    // out(1,0) = input(1,0) + input(2,1) = 4 + 9 = 13
    EXPECT_NEAR(out(0, 0, 1, 0), 13.0f, EPS);
}

TEST(ConvLayerForward, OutputIsRank4) {
    ConvLayer conv(1, 4, 3, 3, 1, 1, 1, 1);
    Tensor input(2, 1, 8, 8);
    input.setRandom();
    EXPECT_EQ(conv.forward(input).ndim(), 4);
}

// =============================================================================
// ── Backward — forme des gradients ───────────────────────────────────────────
// =============================================================================

TEST(ConvLayerBackward, GradInputShape_SamePadding) {
    ConvLayer conv(1, 4, 3, 3, 1, 1, 1, 1);
    Tensor input(2, 1, 8, 8);
    input.setRandom();
    Tensor out = conv.forward(input);

    Tensor grad_out(out.shape());
    grad_out.setConstant(1.0f);
    Tensor grad_in = conv.backward(grad_out);

    EXPECT_EQ(grad_in.dim(0), 2);
    EXPECT_EQ(grad_in.dim(1), 1);
    EXPECT_EQ(grad_in.dim(2), 8);
    EXPECT_EQ(grad_in.dim(3), 8);
}

TEST(ConvLayerBackward, GradInputShape_NoPadding) {
    ConvLayer conv(1, 4, 3, 3, 1, 1, 0, 0);
    Tensor input(2, 1, 8, 8);
    input.setRandom();
    Tensor out = conv.forward(input);

    Tensor grad_out(out.shape());
    grad_out.setConstant(1.0f);
    Tensor grad_in = conv.backward(grad_out);

    // grad_input doit avoir la forme de l'entrée originale
    EXPECT_EQ(grad_in.dim(0), 2);
    EXPECT_EQ(grad_in.dim(1), 1);
    EXPECT_EQ(grad_in.dim(2), 8);
    EXPECT_EQ(grad_in.dim(3), 8);
}

TEST(ConvLayerBackward, GradInputShape_Stride2) {
    ConvLayer conv(1, 4, 3, 3, 2, 2, 1, 1);
    Tensor input(1, 1, 14, 14);
    input.setRandom();
    Tensor out = conv.forward(input);

    Tensor grad_out(out.shape());
    grad_out.setConstant(1.0f);
    Tensor grad_in = conv.backward(grad_out);

    EXPECT_EQ(grad_in.dim(2), 14);
    EXPECT_EQ(grad_in.dim(3), 14);
}

TEST(ConvLayerBackward, GradWeightsShape) {
    ConvLayer conv(2, 8, 3, 3, 1, 1, 1, 1);
    Tensor input(2, 2, 8, 8);
    input.setRandom();
    Tensor out = conv.forward(input);

    Tensor grad_out(out.shape());
    grad_out.setConstant(1.0f);
    conv.backward(grad_out);

    // [out_ch, in_ch, kH, kW]
    const Tensor& gw = conv.getWeightGradients();
    EXPECT_EQ(gw.dim(0), 8);
    EXPECT_EQ(gw.dim(1), 2);
    EXPECT_EQ(gw.dim(2), 3);
    EXPECT_EQ(gw.dim(3), 3);
}

TEST(ConvLayerBackward, GradBiasShape) {
    ConvLayer conv(1, 6, 3, 3, 1, 1, 1, 1);
    Tensor input(2, 1, 8, 8);
    input.setRandom();
    conv.forward(input);

    Tensor grad_out(2, 6, 8, 8);
    grad_out.setConstant(1.0f);
    conv.backward(grad_out);

    EXPECT_EQ(conv.getBiasGradients().size(), 6);
}

// =============================================================================
// ── Backward — valeurs des gradients ─────────────────────────────────────────
// =============================================================================

TEST(ConvLayerBackward, GradWeightsNonZero) {
    ConvLayer conv(1, 4, 3, 3, 1, 1, 1, 1);
    Tensor input(2, 1, 8, 8);
    input.setRandom();
    conv.forward(input);

    Tensor grad_out(2, 4, 8, 8);
    grad_out.setConstant(1.0f);
    conv.backward(grad_out);

    float norm = 0.f;
    const Tensor& gw = conv.getWeightGradients();
    for (int i = 0; i < gw.size(); ++i) norm += gw[i] * gw[i];
    EXPECT_GT(norm, 0.f);
}

TEST(ConvLayerBackward, GradBiasNonZero) {
    ConvLayer conv(1, 4, 3, 3, 1, 1, 1, 1);
    Tensor input(1, 1, 4, 4);
    input.setRandom();
    conv.forward(input);

    Tensor grad_out(1, 4, 4, 4);
    grad_out.setConstant(1.0f);
    conv.backward(grad_out);

    EXPECT_GT(conv.getBiasGradients().norm(), 0.f);
}

TEST(ConvLayerBackward, GradInputNonZero) {
    ConvLayer conv(1, 4, 3, 3, 1, 1, 1, 1);
    Tensor input(1, 1, 8, 8);
    input.setRandom();
    Tensor out = conv.forward(input);

    Tensor grad_out(out.shape());
    grad_out.setConstant(1.0f);
    Tensor grad_in = conv.backward(grad_out);

    float norm = 0.f;
    for (int i = 0; i < grad_in.size(); ++i) norm += grad_in[i] * grad_in[i];
    EXPECT_GT(norm, 0.f);
}

TEST(ConvLayerBackward, ZeroGradOutput_ZeroGradWeights) {
    ConvLayer conv(1, 4, 3, 3, 1, 1, 1, 1);
    Tensor input(1, 1, 6, 6);
    input.setRandom();
    conv.forward(input);

    Tensor grad_out(1, 4, 6, 6);
    grad_out.setZero();
    conv.backward(grad_out);

    const Tensor& gw = conv.getWeightGradients();
    for (int i = 0; i < gw.size(); ++i)
        EXPECT_NEAR(gw[i], 0.0f, EPS);
}

// =============================================================================
// ── Gradient check numérique ──────────────────────────────────────────────────
// Vérifie dL/dW via différences finies centrées sur un cas minimaliste
// =============================================================================

TEST(ConvLayerGradientCheck, WeightGrad_FiniteDiff) {
    // Configuration minimaliste pour que le check soit rapide
    // 1ch→1ch, kernel 2×2, pad 0, stride 1 → 4 poids à vérifier
    const float H_STEP = 1e-3f;
    const float TOL    = 1e-2f;

    ConvLayer conv(1, 1, 2, 2, 1, 1, 0, 0);

    // Poids fixes reproductibles
    Tensor w(1, 1, 2, 2);
    w(0, 0, 0, 0) = 0.5f;  w(0, 0, 0, 1) = -0.3f;
    w(0, 0, 1, 0) = 0.1f;  w(0, 0, 1, 1) =  0.8f;
    conv.setWeights(w);
    conv.setBias(Eigen::VectorXf::Zero(1));

    // Entrée fixe 1×1×3×3
    Tensor input(1, 1, 3, 3);
    for (int i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(i + 1) * 0.1f;

    // Passe forward + backward analytique
    conv.forward(input);
    Tensor grad_out(1, 1, 2, 2);
    grad_out.setConstant(1.0f);
    conv.backward(grad_out);
    const Tensor& analytic_gw = conv.getWeightGradients();

    // Loss = somme de tous les éléments de la sortie
    auto computeLoss = [&](const Tensor& ww) -> float {
        ConvLayer c(1, 1, 2, 2, 1, 1, 0, 0);
        c.setWeights(ww);
        c.setBias(Eigen::VectorXf::Zero(1));
        Tensor out = c.forward(input);
        float sum = 0.f;
        for (int i = 0; i < out.size(); ++i) sum += out[i];
        return sum;
    };

    // Vérification pour chacun des 4 poids
    for (int i = 0; i < w.size(); ++i) {
        Tensor wp = w, wm = w;
        wp[i] += H_STEP;
        wm[i] -= H_STEP;
        float numeric = (computeLoss(wp) - computeLoss(wm)) / (2.0f * H_STEP);
        EXPECT_NEAR(analytic_gw[i], numeric, TOL)
            << "Grad check échoue pour weight[" << i << "]";
    }
}

TEST(ConvLayerGradientCheck, BiasGrad_FiniteDiff) {
    const float H_STEP = 1e-3f;
    const float TOL    = 1e-2f;

    ConvLayer conv(1, 2, 2, 2, 1, 1, 0, 0);

    Tensor w(2, 1, 2, 2);
    w.setConstant(0.1f);
    conv.setWeights(w);
    Eigen::VectorXf b(2);
    b << 0.3f, -0.2f;
    conv.setBias(b);

    Tensor input(1, 1, 3, 3);
    for (int i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i) * 0.1f;

    conv.forward(input);
    Tensor grad_out(1, 2, 2, 2);
    grad_out.setConstant(1.0f);
    conv.backward(grad_out);
    const Eigen::VectorXf& analytic_gb = conv.getBiasGradients();

    auto computeLoss = [&](const Eigen::VectorXf& bb) -> float {
        ConvLayer c(1, 2, 2, 2, 1, 1, 0, 0);
        c.setWeights(w);
        c.setBias(bb);
        Tensor out = c.forward(input);
        float sum = 0.f;
        for (int i = 0; i < out.size(); ++i) sum += out[i];
        return sum;
    };

    for (int i = 0; i < b.size(); ++i) {
        Eigen::VectorXf bp = b, bm = b;
        bp(i) += H_STEP;
        bm(i) -= H_STEP;
        float numeric  = (computeLoss(bp) - computeLoss(bm)) / (2.0f * H_STEP);
        EXPECT_NEAR(analytic_gb(i), numeric, TOL)
            << "Bias grad check échoue pour bias[" << i << "]";
    }
}

// =============================================================================
// ── updateParams ─────────────────────────────────────────────────────────────
// =============================================================================

TEST(ConvLayerUpdateParams, SGD_WeightsChange) {
    ConvLayer conv(1, 2, 3, 3, 1, 1, 1, 1);

    // Snapshot des poids avant
    Tensor w_before = conv.getWeights();

    Tensor input(1, 1, 5, 5);
    input.setRandom();
    conv.forward(input);

    Tensor grad_out(1, 2, 5, 5);
    grad_out.setConstant(0.1f);
    conv.backward(grad_out);

    SGD sgd(0.01f, 0.0f);
    conv.updateParams(sgd);

    bool changed = false;
    const Tensor& w_after = conv.getWeights();
    for (int i = 0; i < w_before.size(); ++i)
        if (std::abs(w_after[i] - w_before[i]) > 1e-8f) { changed = true; break; }
    EXPECT_TRUE(changed);
}

TEST(ConvLayerUpdateParams, Adam_WeightsChange) {
    ConvLayer conv(1, 2, 3, 3, 1, 1, 1, 1);
    Tensor w_before = conv.getWeights();

    Tensor input(1, 1, 5, 5);
    input.setRandom();
    conv.forward(input);

    Tensor grad_out(1, 2, 5, 5);
    grad_out.setConstant(0.1f);
    conv.backward(grad_out);

    Adam adam(0.001f);
    conv.updateParams(adam);

    bool changed = false;
    const Tensor& w_after = conv.getWeights();
    for (int i = 0; i < w_before.size(); ++i)
        if (std::abs(w_after[i] - w_before[i]) > 1e-8f) { changed = true; break; }
    EXPECT_TRUE(changed);
}

TEST(ConvLayerUpdateParams, GradsClearedAfterUpdate) {
    ConvLayer conv(1, 2, 3, 3, 1, 1, 1, 1);
    Tensor input(1, 1, 5, 5);
    input.setRandom();
    conv.forward(input);

    Tensor grad_out(1, 2, 5, 5);
    grad_out.setConstant(1.0f);
    conv.backward(grad_out);

    SGD sgd(0.01f, 0.0f);
    conv.updateParams(sgd);

    // Après updateParams, grad_weights_ doit être remis à zéro
    const Tensor& gw = conv.getWeightGradients();
    for (int i = 0; i < gw.size(); ++i)
        EXPECT_FLOAT_EQ(gw[i], 0.0f);

    EXPECT_FLOAT_EQ(conv.getBiasGradients().norm(), 0.0f);
}

TEST(ConvLayerUpdateParams, MultipleSteps_LossDirection) {
    // Vérifie qu'en entraînant sur une cible constante,
    // la sortie évolue dans le bon sens (vers la cible)
    ConvLayer conv(1, 1, 1, 1); // kernel 1×1 = multiplication scalaire
    Tensor w(1, 1, 1, 1);
    w.setConstant(0.0f);
    conv.setWeights(w);
    conv.setBias(Eigen::VectorXf::Zero(1));

    Tensor input = makeConst(1, 1, 1, 1, 1.0f);
    SGD sgd(0.1f, 0.0f);

    // Cible = 1.0, on minimise (out - 1)² manuellement
    for (int step = 0; step < 20; ++step) {
        Tensor out = conv.forward(input);
        // grad MSE = (out - target) / B
        Tensor grad_out(1, 1, 1, 1);
        grad_out(0, 0, 0, 0) = out(0, 0, 0, 0) - 1.0f;
        conv.backward(grad_out);
        conv.updateParams(sgd);
    }

    // Après 20 étapes, la sortie doit s'être rapprochée de 1.0
    Tensor final_out = conv.forward(input);
    EXPECT_NEAR(final_out(0, 0, 0, 0), 1.0f, 0.1f);
}