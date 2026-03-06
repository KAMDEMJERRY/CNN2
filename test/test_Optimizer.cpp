// tests/test_Optimizer.cpp

#include <gtest/gtest.h>
#include "Optimizer.hpp"
#include <cmath>

static constexpr float EPS = 1e-5f;

// =============================================================================
// ── GradientUtils ─────────────────────────────────────────────────────────────
// =============================================================================

TEST(GradientUtils, ClipTensor_NormReduced) {
    Tensor g(1, 1, 1, 4);
    g[0] = 3.0f; g[1] = 4.0f; g[2] = 0.0f; g[3] = 0.0f; // norm = 5
    GradientUtils::clipByNorm(g, 2.5f);
    float norm = 0.f;
    for (int i = 0; i < g.size(); ++i) norm += g[i] * g[i];
    EXPECT_NEAR(std::sqrt(norm), 2.5f, 1e-4f);
}

TEST(GradientUtils, ClipTensor_NoClip_WhenNormBelow) {
    Tensor g(1, 1, 1, 2);
    g[0] = 1.0f; g[1] = 0.0f; // norm = 1, max = 5
    float v0 = g[0], v1 = g[1];
    GradientUtils::clipByNorm(g, 5.0f);
    EXPECT_NEAR(g[0], v0, EPS);
    EXPECT_NEAR(g[1], v1, EPS);
}

TEST(GradientUtils, ClipMatrix_NormReduced) {
    Eigen::MatrixXf m(2, 2);
    m << 3.0f, 4.0f, 0.0f, 0.0f; // norm = 5
    GradientUtils::clipByNorm(m, 2.5f);
    EXPECT_NEAR(m.norm(), 2.5f, 1e-4f);
}

TEST(GradientUtils, ClipVector_NormReduced) {
    Eigen::VectorXf v(2);
    v << 3.0f, 4.0f; // norm = 5
    GradientUtils::clipByNorm(v, 2.5f);
    EXPECT_NEAR(v.norm(), 2.5f, 1e-4f);
}

TEST(GradientUtils, ClipZeroMax_NoOp) {
    Tensor g(1, 1, 1, 2);
    g[0] = 10.0f; g[1] = 10.0f;
    GradientUtils::clipByNorm(g, 0.0f); // désactivé
    EXPECT_NEAR(g[0], 10.0f, EPS);
}

// =============================================================================
// ── SGD ──────────────────────────────────────────────────────────────────────
// =============================================================================

TEST(SGD, UpdateTensor_WeightsChange) {
    SGD sgd(0.1f, 0.0f);
    Tensor w(1, 1, 1, 4);
    w.setConstant(1.0f);
    Tensor g(1, 1, 1, 4);
    g.setConstant(1.0f);

    sgd.updateWeights(w, g);

    for (int i = 0; i < w.size(); ++i)
        EXPECT_NEAR(w[i], 0.9f, EPS); // w -= lr * g
}

TEST(SGD, UpdateMatrix_WeightsChange) {
    SGD sgd(0.01f, 0.0f);
    Eigen::MatrixXf W = Eigen::MatrixXf::Ones(3, 4);
    Eigen::MatrixXf G = Eigen::MatrixXf::Ones(3, 4);
    sgd.updateWeights(W, G);
    EXPECT_NEAR(W.mean(), 0.99f, 1e-4f);
}

TEST(SGD, UpdateBias_Change) {
    SGD sgd(0.1f, 0.0f);
    Eigen::VectorXf b(3);
    b.setConstant(2.0f);
    Eigen::VectorXf g(3);
    g.setConstant(1.0f);
    sgd.updateBias(b, g);
    EXPECT_NEAR(b(0), 1.9f, EPS);
}

TEST(SGD, UpdateBias_SizeMismatch_Throws) {
    SGD sgd(0.1f);
    Eigen::VectorXf b(3), g(2);
    b.setZero(); g.setZero();
    EXPECT_THROW(sgd.updateBias(b, g), std::runtime_error);
}

TEST(SGD, MomentumAccumulates) {
    // Avec momentum > 0, le pas devient de plus en plus grand
    SGD sgd(0.1f, 0.9f);
    Tensor w(1, 1, 1, 1);
    w.setConstant(5.0f);
    Tensor g(1, 1, 1, 1);
    g.setConstant(1.0f);

    float prev_w = w[0];
    float first_step, second_step;

    sgd.updateWeights(w, g);
    first_step = std::abs(w[0] - prev_w);

    prev_w = w[0];
    sgd.updateWeights(w, g);
    second_step = std::abs(w[0] - prev_w);

    // La deuxième mise à jour doit être plus grande (momentum accumulé)
    EXPECT_GT(second_step, first_step);
}

TEST(SGD, GetSetLearningRate) {
    SGD sgd(0.01f);
    EXPECT_NEAR(sgd.getLearningRate(), 0.01f, EPS);
    sgd.setLearningRate(0.1f);
    EXPECT_NEAR(sgd.getLearningRate(), 0.1f, EPS);
}

// =============================================================================
// ── Adam ─────────────────────────────────────────────────────────────────────
// =============================================================================

TEST(Adam, UpdateTensor_WeightsChange) {
    Adam adam(0.001f);
    Tensor w(1, 1, 1, 4);
    w.setConstant(1.0f);
    Tensor g(1, 1, 1, 4);
    g.setConstant(1.0f);
    adam.updateWeights(w, g);

    // Adam avec lr=0.001 et gradient=1 doit changer les poids
    bool changed = false;
    for (int i = 0; i < w.size(); ++i)
        if (std::abs(w[i] - 1.0f) > 1e-8f) { changed = true; break; }
    EXPECT_TRUE(changed);
}

TEST(Adam, UpdateMatrix_WeightsChange) {
    Adam adam(0.001f);
    Eigen::MatrixXf W = Eigen::MatrixXf::Ones(2, 3);
    Eigen::MatrixXf G = Eigen::MatrixXf::Ones(2, 3);
    adam.updateWeights(W, G);
    EXPECT_LT(W.mean(), 1.0f); // poids réduits
}

TEST(Adam, UpdateBias_Change) {
    Adam adam(0.001f);
    Eigen::VectorXf b(4);
    b.setConstant(1.0f);
    Eigen::VectorXf g(4);
    g.setConstant(1.0f);
    adam.updateBias(b, g);
    EXPECT_LT(b(0), 1.0f);
}

TEST(Adam, MultipleSteps_Converges) {
    // Adam doit réduire un paramètre scalaire vers zéro en minimisant l =  w * w
    Adam adam(0.1f);
    Tensor w(1, 1, 1, 1);
    w.setConstant(2.0f);

    for (int i = 0; i < 100; ++i) {
        Tensor g(1, 1, 1, 1);
        g.setConstant(w[0]); // gradient de (1/2)*w^2 est w
        adam.updateWeights(w, g);
    }
    EXPECT_LT(std::abs(w[0]), 0.1f);
}

// =============================================================================
// ── StepDecay ────────────────────────────────────────────────────────────────
// =============================================================================

TEST(StepDecay, EpochZero_InitialLR) {
    StepDecay sched(0.01f, 0.5f, 5);
    EXPECT_NEAR(sched.getLR(0), 0.01f, EPS);
}

TEST(StepDecay, AfterOneDrop) {
    StepDecay sched(0.01f, 0.5f, 5);
    // Epoch 5 → floor(5/5)=1 → lr = 0.01 * 0.5 = 0.005
    EXPECT_NEAR(sched.getLR(5), 0.005f, EPS);
}

TEST(StepDecay, AfterTwoDrops) {
    StepDecay sched(0.01f, 0.5f, 5);
    // Epoch 10 → floor(10/5)=2 → lr = 0.01 * 0.25 = 0.0025
    EXPECT_NEAR(sched.getLR(10), 0.0025f, EPS);
}

TEST(StepDecay, Apply_ModifiesOptimizer) {
    StepDecay sched(0.01f, 0.5f, 5);
    SGD sgd(0.01f);
    sched.apply(sgd, 5);
    EXPECT_NEAR(sgd.getLearningRate(), 0.005f, EPS);
}
