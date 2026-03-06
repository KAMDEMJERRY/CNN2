// tests/test_DenseLayer.cpp

#include <gtest/gtest.h>
#include "DenseLayer.hpp"
#include "Optimizer.hpp"
#include <cmath>

static constexpr float EPS = 1e-4f;

// =============================================================================
// ── Construction ─────────────────────────────────────────────────────────────
// =============================================================================

TEST(DenseCtor, Dimensions) {
    DenseLayer dense(12, 4);
    EXPECT_EQ(dense.getInputSize(),  12);
    EXPECT_EQ(dense.getOutputSize(), 4);
}

TEST(DenseCtor, WeightsShape) {
    DenseLayer dense(8, 3);
    EXPECT_EQ(dense.getWeights().rows(), 3);
    EXPECT_EQ(dense.getWeights().cols(), 8);
}

TEST(DenseCtor, BiasShape) {
    DenseLayer dense(8, 5);
    EXPECT_EQ(dense.getBias().size(), 5);
}

TEST(DenseCtor, BiasInitializedToZero) {
    DenseLayer dense(4, 3);
    for (int i = 0; i < dense.getBias().size(); ++i)
        EXPECT_FLOAT_EQ(dense.getBias()(i), 0.0f);
}

TEST(DenseCtor, GetName) {
    DenseLayer dense(4, 2);
    EXPECT_EQ(dense.getName(), "Dense");
}

TEST(DenseCtor, InitXavier_WeightsNonZero) {
    DenseLayer dense(16, 8);
    dense.initializeWeights("xavier");
    EXPECT_GT(dense.getWeights().norm(), 0.0f);
}

TEST(DenseCtor, InitHe_WeightsNonZero) {
    DenseLayer dense(16, 8);
    dense.initializeWeights("he");
    EXPECT_GT(dense.getWeights().norm(), 0.0f);
}

// =============================================================================
// ── Forward ──────────────────────────────────────────────────────────────────
// =============================================================================

TEST(DenseForward, OutputShape_4D) {
    DenseLayer dense(6, 4);
    // Entrée (B=2, C=2, H=1, W=3) → aplatie en (2, 6)
    Tensor input(2, 2, 1, 3);
    input.setRandom();
    Tensor out = dense.forward(input);

    EXPECT_EQ(out.dim(0), 2);
    EXPECT_EQ(out.dim(1), 4);
    EXPECT_EQ(out.dim(2), 1);
    EXPECT_EQ(out.dim(3), 1);
}

TEST(DenseForward, OutputIsRank4) {
    DenseLayer dense(4, 2);
    Tensor input(1, 1, 2, 2);
    input.setRandom();
    EXPECT_EQ(dense.forward(input).ndim(), 4);
}

TEST(DenseForward, ZeroWeights_OutputIsBias) {
    DenseLayer dense(4, 2);
    dense.getWeights().setZero();
    Eigen::VectorXf b(2);
    b << 3.0f, -1.0f;
    dense.getBias() = b;

    Tensor input(1, 1, 2, 2);
    input.setConstant(99.0f);
    Tensor out = dense.forward(input);

    EXPECT_NEAR(out(0, 0, 0, 0),  3.0f, EPS);
    EXPECT_NEAR(out(0, 1, 0, 0), -1.0f, EPS);
}

TEST(DenseForward, IdentityWeight_SingleInput) {
    // W = I (2x2), b=0, entrée [1,2] → sortie [1,2]
    DenseLayer dense(2, 2);
    Eigen::MatrixXf W = Eigen::MatrixXf::Identity(2, 2);
    dense.setWeights(W);
    dense.setBias(Eigen::VectorXf::Zero(2));

    Tensor input(1, 1, 1, 2);
    input.setZero();
    input[0] = 1.0f;
    input[1] = 2.0f;

    Tensor out = dense.forward(input);
    EXPECT_NEAR(out(0, 0, 0, 0), 1.0f, EPS);
    EXPECT_NEAR(out(0, 1, 0, 0), 2.0f, EPS);
}

TEST(DenseForward, BatchPreserved) {
    DenseLayer dense(4, 3);
    for (int B : {1, 4, 8}) {
        Tensor input(B, 1, 2, 2);
        input.setRandom();
        EXPECT_EQ(dense.forward(input).dim(0), B);
    }
}

// =============================================================================
// ── Backward — formes ────────────────────────────────────────────────────────
// =============================================================================

TEST(DenseBackward, GradInputShape) {
    DenseLayer dense(6, 4);
    Tensor input(2, 2, 1, 3);
    input.setRandom();
    dense.forward(input);

    Tensor grad_out(2, 4, 1, 1);
    grad_out.setConstant(1.0f);
    Tensor grad_in = dense.backward(grad_out);

    EXPECT_EQ(grad_in.dim(0), 2);
    EXPECT_EQ(grad_in.size(), input.size());
}

TEST(DenseBackward, GradWeightsShape) {
    DenseLayer dense(6, 4);
    Tensor input(2, 2, 1, 3);
    input.setRandom();
    dense.forward(input);

    Tensor grad_out(2, 4, 1, 1);
    grad_out.setConstant(1.0f);
    dense.backward(grad_out);

    EXPECT_EQ(dense.getWeightGradients().rows(), 4);
    EXPECT_EQ(dense.getWeightGradients().cols(), 6);
}

TEST(DenseBackward, GradBiasShape) {
    DenseLayer dense(4, 3);
    Tensor input(2, 1, 2, 2);
    input.setRandom();
    dense.forward(input);

    Tensor grad_out(2, 3, 1, 1);
    grad_out.setConstant(1.0f);
    dense.backward(grad_out);

    EXPECT_EQ(dense.getBiasGradients().size(), 3);
}

TEST(DenseBackward, GradWeightsNonZero) {
    DenseLayer dense(4, 3);
    Tensor input(2, 1, 2, 2);
    input.setRandom();
    dense.forward(input);

    Tensor grad_out(2, 3, 1, 1);
    grad_out.setConstant(1.0f);
    dense.backward(grad_out);

    EXPECT_GT(dense.getWeightGradients().norm(), 0.0f);
}

TEST(DenseBackward, GradBiasNonZero) {
    DenseLayer dense(4, 3);
    Tensor input(2, 1, 2, 2);
    input.setRandom();
    dense.forward(input);

    Tensor grad_out(2, 3, 1, 1);
    grad_out.setConstant(1.0f);
    dense.backward(grad_out);

    EXPECT_GT(dense.getBiasGradients().norm(), 0.0f);
}

// =============================================================================
// ── GradientCheck numérique ───────────────────────────────────────────────────
// =============================================================================

TEST(DenseGradCheck, WeightGrad_FiniteDiff) {
    const float H_STEP = 1e-3f;
    const float TOL    = 1e-2f;

    DenseLayer dense(4, 2);
    dense.getWeights().setConstant(0.1f);
    dense.getBias().setZero();

    Tensor input(1, 1, 2, 2);
    for (int i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i + 1) * 0.2f;

    dense.forward(input);
    Tensor grad_out(1, 2, 1, 1);
    grad_out.setConstant(1.0f);
    dense.backward(grad_out);
    const Eigen::MatrixXf analytic = dense.getWeightGradients();

    auto computeLoss = [&](const Eigen::MatrixXf& W) -> float {
        DenseLayer c(4, 2);
        c.setWeights(W);
        c.setBias(Eigen::VectorXf::Zero(2));
        Tensor out = c.forward(input);
        float s = 0.f;
        for (int i = 0; i < out.size(); ++i) s += out[i];
        return s;
    };

    Eigen::MatrixXf W = dense.getWeights();
    for (int r = 0; r < W.rows(); ++r) {
        for (int c_ = 0; c_ < W.cols(); ++c_) {
            Eigen::MatrixXf Wp = W, Wm = W;
            Wp(r, c_) += H_STEP;
            Wm(r, c_) -= H_STEP;
            float numeric = (computeLoss(Wp) - computeLoss(Wm)) / (2.0f * H_STEP);
            EXPECT_NEAR(analytic(r, c_), numeric, TOL)
                << "Grad check échoue pour W[" << r << "," << c_ << "]";
        }
    }
}

TEST(DenseGradCheck, BiasGrad_FiniteDiff) {
    const float H_STEP = 1e-3f;
    const float TOL    = 1e-2f;

    DenseLayer dense(4, 2);
    dense.getWeights().setConstant(0.1f);
    Eigen::VectorXf b(2);
    b << 0.5f, -0.3f;
    dense.getBias() = b;

    Tensor input(1, 1, 2, 2);
    for (int i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i + 1) * 0.1f;

    dense.forward(input);
    Tensor grad_out(1, 2, 1, 1);
    grad_out.setConstant(1.0f);
    dense.backward(grad_out);
    const Eigen::VectorXf analytic = dense.getBiasGradients();

    auto computeLoss = [&](const Eigen::VectorXf& bb) -> float {
        DenseLayer c(4, 2);
        c.setWeights(dense.getWeights());
        c.setBias(bb);
        Tensor out = c.forward(input);
        float s = 0.f;
        for (int i = 0; i < out.size(); ++i) s += out[i];
        return s;
    };

    for (int i = 0; i < b.size(); ++i) {
        Eigen::VectorXf bp = b, bm = b;
        bp(i) += H_STEP;
        bm(i) -= H_STEP;
        float numeric = (computeLoss(bp) - computeLoss(bm)) / (2.0f * H_STEP);
        EXPECT_NEAR(analytic(i), numeric, TOL)
            << "Bias grad check échoue pour b[" << i << "]";
    }
}

// =============================================================================
// ── updateParams ─────────────────────────────────────────────────────────────
// =============================================================================

TEST(DenseUpdateParams, SGD_WeightsChange) {
    DenseLayer dense(4, 3);
    dense.initializeWeights("xavier");
    Eigen::MatrixXf W_before = dense.getWeights();

    Tensor input(1, 1, 2, 2);
    input.setRandom();
    dense.forward(input);

    Tensor grad_out(1, 3, 1, 1);
    grad_out.setConstant(0.5f);
    dense.backward(grad_out);

    SGD sgd(0.01f, 0.0f);
    dense.updateParams(sgd);

    EXPECT_GT((dense.getWeights() - W_before).norm(), 1e-8f);
}

TEST(DenseUpdateParams, Adam_WeightsChange) {
    DenseLayer dense(4, 3);
    dense.initializeWeights("xavier");
    Eigen::MatrixXf W_before = dense.getWeights();

    Tensor input(1, 1, 2, 2);
    input.setRandom();
    dense.forward(input);

    Tensor grad_out(1, 3, 1, 1);
    grad_out.setConstant(0.5f);
    dense.backward(grad_out);

    Adam adam(0.001f);
    dense.updateParams(adam);

    EXPECT_GT((dense.getWeights() - W_before).norm(), 1e-8f);
}

TEST(DenseUpdateParams, GradsClearedAfterUpdate) {
    DenseLayer dense(4, 3);
    Tensor input(1, 1, 2, 2);
    input.setRandom();
    dense.forward(input);

    Tensor grad_out(1, 3, 1, 1);
    grad_out.setConstant(1.0f);
    dense.backward(grad_out);

    SGD sgd(0.01f, 0.0f);
    dense.updateParams(sgd);

    EXPECT_FLOAT_EQ(dense.getWeightGradients().norm(), 0.0f);
    EXPECT_FLOAT_EQ(dense.getBiasGradients().norm(), 0.0f);
}

TEST(DenseUpdateParams, SGD_LossDecreases) {
    // Minimise y = Wx avec cible constante → poids convergent vers cible
    DenseLayer dense(1, 1);
    dense.getWeights().setConstant(0.0f);
    dense.getBias().setZero();

    Tensor input(1, 1, 1, 1);
    input.setConstant(1.0f);
    SGD sgd(0.1f, 0.0f);

    // On cible out = 1 via MSE
    for (int step = 0; step < 30; ++step) {
        Tensor out = dense.forward(input);
        Tensor grad_out(1, 1, 1, 1);
        grad_out[0] = out[0] - 1.0f;
        dense.backward(grad_out);
        dense.updateParams(sgd);
    }

    Tensor final_out = dense.forward(input);
    EXPECT_NEAR(final_out[0], 1.0f, 0.1f);
}
