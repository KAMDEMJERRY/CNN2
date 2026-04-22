// tests/test_DenseLayerModelParallel.cpp
//
// Suite de tests pour DenseLayerModelParallel.
// Vérifie :
//   1. Formes des sorties (forward et backward shapes)
//   2. Cohérence numérique avec DenseLayer (mêmes W, b → même Y)
//   3. Gradient check numérique (différences finies)

#include <gtest/gtest.h>
#include "DenseLayerModelParallel.hpp"
#include "DenseLayer.hpp"
#include "Optimizer.hpp"
#include <cmath>

static constexpr float EPS  = 1e-4f;
static constexpr float GTOL = 1e-2f;  // tolérance gradient numérique

// =============================================================================
// ── Construction ─────────────────────────────────────────────────────────────
// =============================================================================

TEST(ModelParallelCtor, Dimensions) {
    DenseLayerModelParallel layer(12, 4, 4);
    EXPECT_EQ(layer.getInputSize(),  12);
    EXPECT_EQ(layer.getOutputSize(), 4);
}

TEST(ModelParallelCtor, GetName) {
    DenseLayerModelParallel layer(4, 2, 2);
    EXPECT_EQ(layer.getName(), "DenseModelParallel");
}

TEST(ModelParallelCtor, WeightsShape) {
    DenseLayerModelParallel layer(8, 3, 2);
    EXPECT_EQ(layer.getWeights().rows(), 3);
    EXPECT_EQ(layer.getWeights().cols(), 8);
}

// =============================================================================
// ── Forward — formes ─────────────────────────────────────────────────────────
// =============================================================================

TEST(ModelParallelForward, OutputShape_4D) {
    DenseLayerModelParallel layer(6, 4, 2);
    Tensor input(2, 2, 1, 3);
    input.setRandom();
    Tensor out = layer.forward(input);

    EXPECT_EQ(out.dim(0), 2);
    EXPECT_EQ(out.dim(1), 4);
    EXPECT_EQ(out.dim(2), 1);
    EXPECT_EQ(out.dim(3), 1);
}

TEST(ModelParallelForward, OutputIsRank4) {
    DenseLayerModelParallel layer(4, 2, 2);
    Tensor input(1, 1, 2, 2);
    input.setRandom();
    EXPECT_EQ(layer.forward(input).ndim(), 4);
}

TEST(ModelParallelForward, BatchPreserved) {
    DenseLayerModelParallel layer(4, 3, 4);
    for (int B : {1, 4, 8}) {
        Tensor input(B, 1, 2, 2);
        input.setRandom();
        EXPECT_EQ(layer.forward(input).dim(0), B);
    }
}

// =============================================================================
// ── Cohérence avec DenseLayer (même W, b → même sortie) ─────────────────────
// =============================================================================

TEST(ModelParallelForward, SameResultAsDenseLayer) {
    const int Din = 6, Dout = 4;
    DenseLayer          ref(Din, Dout);
    DenseLayerModelParallel par(Din, Dout, 2);

    // Imposer les mêmes poids et biais
    ref.getWeights().setRandom();
    par.setWeights(ref.getWeights());
    ref.getBias().setRandom();
    par.setBias(ref.getBias());

    Tensor input(2, 2, 1, 3);
    input.setRandom();

    Tensor out_ref = ref.forward(input);
    Tensor out_par = par.forward(input);

    ASSERT_EQ(out_ref.size(), out_par.size());
    for (int i = 0; i < out_ref.size(); ++i)
        EXPECT_NEAR(out_ref[i], out_par[i], EPS)
            << "Différence à l'indice " << i;
}

TEST(ModelParallelBackward, GradInputMatchesDenseLayer) {
    const int Din = 6, Dout = 4;
    DenseLayer          ref(Din, Dout);
    DenseLayerModelParallel par(Din, Dout, 2);

    ref.getWeights().setRandom();
    par.setWeights(ref.getWeights());
    ref.getBias().setZero();
    par.setBias(Eigen::VectorXf::Zero(Dout));

    Tensor input(2, 2, 1, 3);
    input.setRandom();

    ref.forward(input);
    par.forward(input);

    Tensor grad_out(2, Dout, 1, 1);
    grad_out.setRandom();

    Tensor dX_ref = ref.backward(grad_out);
    Tensor dX_par = par.backward(grad_out);

    ASSERT_EQ(dX_ref.size(), dX_par.size());
    for (int i = 0; i < dX_ref.size(); ++i)
        EXPECT_NEAR(dX_ref[i], dX_par[i], EPS)
            << "dX différence à l'indice " << i;
}

TEST(ModelParallelBackward, GradWeightsMatchesDenseLayer) {
    const int Din = 6, Dout = 4;
    DenseLayer          ref(Din, Dout);
    DenseLayerModelParallel par(Din, Dout, 2);

    ref.getWeights().setRandom();
    par.setWeights(ref.getWeights());
    ref.getBias().setZero();
    par.setBias(Eigen::VectorXf::Zero(Dout));

    Tensor input(3, 2, 1, 3);
    input.setRandom();

    ref.forward(input);
    par.forward(input);

    Tensor grad_out(3, Dout, 1, 1);
    grad_out.setRandom();

    ref.backward(grad_out);
    par.backward(grad_out);

    ASSERT_EQ(ref.getWeightGradients().rows(), par.getWeightGradients().rows());
    for (int r = 0; r < ref.getWeightGradients().rows(); ++r)
        for (int c = 0; c < ref.getWeightGradients().cols(); ++c)
            EXPECT_NEAR(ref.getWeightGradients()(r,c),
                        par.getWeightGradients()(r,c), EPS)
                << "grad_W diff à [" << r << "," << c << "]";
}

// =============================================================================
// ── Gradient check numérique ──────────────────────────────────────────────────
// =============================================================================

TEST(ModelParallelGradCheck, WeightGrad_FiniteDiff) {
    const float H = 1e-3f;

    DenseLayerModelParallel layer(4, 2, 2);
    layer.getWeights().setConstant(0.1f);
    layer.getBias().setZero();

    Tensor input(1, 1, 2, 2);
    for (int i = 0; i < input.size(); ++i) input[i] = (i + 1) * 0.2f;

    layer.forward(input);
    Tensor grad_out(1, 2, 1, 1);
    grad_out.setConstant(1.0f);
    layer.backward(grad_out);
    Eigen::MatrixXf analytic = layer.getWeightGradients();

    auto loss = [&](const Eigen::MatrixXf& W) -> float {
        DenseLayerModelParallel c(4, 2, 2);
        c.setWeights(W);
        c.setBias(Eigen::VectorXf::Zero(2));
        Tensor out = c.forward(input);
        float s = 0.f;
        for (int i = 0; i < out.size(); ++i) s += out[i];
        return s;
    };

    Eigen::MatrixXf W = layer.getWeights();
    for (int r = 0; r < W.rows(); ++r)
        for (int c = 0; c < W.cols(); ++c) {
            Eigen::MatrixXf Wp = W, Wm = W;
            Wp(r,c) += H; Wm(r,c) -= H;
            float num = (loss(Wp) - loss(Wm)) / (2.f * H);
            EXPECT_NEAR(analytic(r,c), num, GTOL)
                << "Grad W[" << r << "," << c << "]";
        }
}

TEST(ModelParallelGradCheck, BiasGrad_FiniteDiff) {
    const float H = 1e-3f;

    DenseLayerModelParallel layer(4, 2, 2);
    layer.getWeights().setConstant(0.1f);
    Eigen::VectorXf b(2); b << 0.5f, -0.3f;
    layer.getBias() = b;

    Tensor input(1, 1, 2, 2);
    for (int i = 0; i < input.size(); ++i) input[i] = (i + 1) * 0.1f;

    layer.forward(input);
    Tensor grad_out(1, 2, 1, 1);
    grad_out.setConstant(1.0f);
    layer.backward(grad_out);
    Eigen::VectorXf analytic = layer.getBiasGradients();

    auto loss = [&](const Eigen::VectorXf& bb) -> float {
        DenseLayerModelParallel c(4, 2, 2);
        c.setWeights(layer.getWeights());
        c.setBias(bb);
        Tensor out = c.forward(input);
        float s = 0.f;
        for (int i = 0; i < out.size(); ++i) s += out[i];
        return s;
    };

    for (int i = 0; i < b.size(); ++i) {
        Eigen::VectorXf bp = b, bm = b;
        bp(i) += H; bm(i) -= H;
        float num = (loss(bp) - loss(bm)) / (2.f * H);
        EXPECT_NEAR(analytic(i), num, GTOL)
            << "Grad bias[" << i << "]";
    }
}

// =============================================================================
// ── updateParams ─────────────────────────────────────────────────────────────
// =============================================================================

TEST(ModelParallelUpdate, GradsClearedAfterUpdate) {
    DenseLayerModelParallel layer(4, 3, 2);
    Tensor input(1, 1, 2, 2);
    input.setRandom();
    layer.forward(input);

    Tensor grad_out(1, 3, 1, 1);
    grad_out.setConstant(1.0f);
    layer.backward(grad_out);

    SGD sgd(0.01f, 0.0f);
    layer.updateParams(sgd);

    EXPECT_FLOAT_EQ(layer.getWeightGradients().norm(), 0.0f);
    EXPECT_FLOAT_EQ(layer.getBiasGradients().norm(), 0.0f);
}
