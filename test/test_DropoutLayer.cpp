// tests/test_DropoutLayer.cpp

#include <gtest/gtest.h>
#include "DropoutLayer.hpp"
#include <cmath>
#include <numeric>

static constexpr float EPS = 1e-5f;

// =============================================================================
// ── Construction ─────────────────────────────────────────────────────────────
// =============================================================================

TEST(DropoutCtor, DefaultRate) {
    DropoutLayer d(0.5f);
    EXPECT_NEAR(d.getRate(), 0.5f, EPS);
}

TEST(DropoutCtor, InvalidRate_Zero_Throws) {
    EXPECT_THROW(DropoutLayer(0.0f), std::invalid_argument);
}

TEST(DropoutCtor, InvalidRate_One_Throws) {
    EXPECT_THROW(DropoutLayer(1.0f), std::invalid_argument);
}

TEST(DropoutCtor, InvalidRate_Negative_Throws) {
    EXPECT_THROW(DropoutLayer(-0.1f), std::invalid_argument);
}

TEST(DropoutCtor, GetName) {
    DropoutLayer d(0.3f);
    EXPECT_EQ(d.getName(), "DropoutLayer");
}

// =============================================================================
// ── Mode évaluation ──────────────────────────────────────────────────────────
// =============================================================================

TEST(DropoutForward, EvalMode_IdentityPassthrough) {
    DropoutLayer d(0.5f);
    d.eval(); // mode inférence → pas de dropout
    Tensor input(1, 1, 4, 4);
    input.setRandom();
    Tensor out = d.forward(input);
    for (int i = 0; i < input.size(); ++i)
        EXPECT_NEAR(out[i], input[i], EPS);
}

TEST(DropoutForward, EvalMode_GetName) {
    DropoutLayer d(0.5f);
    d.eval();
    EXPECT_EQ(d.getName(), "DropoutLayer");
}

// =============================================================================
// ── Mode entraînement ────────────────────────────────────────────────────────
// =============================================================================

TEST(DropoutForward, TrainMode_SomeZeros) {
    // Avec un grand tenseur et rate=0.5, il doit y avoir des zéros
    DropoutLayer d(0.5f);
    d.train();
    Tensor input(1, 1, 20, 20); // 400 éléments
    input.setConstant(1.0f);
    Tensor out = d.forward(input);

    int zeros = 0;
    for (int i = 0; i < out.size(); ++i)
        if (out[i] == 0.0f) ++zeros;

    // Statistiquement, ~50% doivent être à 0 (on tolère une large fenêtre)
    EXPECT_GT(zeros, 50);
    EXPECT_LT(zeros, 350);
}

TEST(DropoutForward, TrainMode_NonZeroScaled) {
    // Les valeurs non-nulles doivent être multipliées par 1/(1-rate)
    DropoutLayer d(0.5f);
    d.train();
    const float expected_scale = 1.0f / (1.0f - 0.5f); // = 2.0

    Tensor input(1, 1, 10, 10);
    input.setConstant(1.0f);
    Tensor out = d.forward(input);

    for (int i = 0; i < out.size(); ++i) {
        if (out[i] != 0.0f) {
            EXPECT_NEAR(out[i], expected_scale, EPS);
        }
    }
}

TEST(DropoutForward, TrainMode_ShapePreserved) {
    DropoutLayer d(0.3f);
    d.train();
    Tensor input(2, 4, 8, 8);
    input.setRandom();
    Tensor out = d.forward(input);
    EXPECT_EQ(out.dim(0), 2);
    EXPECT_EQ(out.dim(1), 4);
    EXPECT_EQ(out.dim(2), 8);
    EXPECT_EQ(out.dim(3), 8);
}

TEST(DropoutForward, HighRate_MostlyZeros) {
    DropoutLayer d(0.9f); // 90% dropout
    d.train();
    Tensor input(1, 1, 50, 50); // 2500 éléments
    input.setConstant(1.0f);
    Tensor out = d.forward(input);

    int zeros = 0;
    for (int i = 0; i < out.size(); ++i)
        if (out[i] == 0.0f) ++zeros;

    // On s'attend à ~90% de zéros [>= 2000]
    EXPECT_GT(zeros, 1500);
}

// =============================================================================
// ── Backward ─────────────────────────────────────────────────────────────────
// =============================================================================

TEST(DropoutBackward, EvalMode_GradPassthrough) {
    DropoutLayer d(0.5f);
    d.eval();
    Tensor input(1, 1, 3, 3);
    input.setRandom();
    d.forward(input);

    Tensor grad_out(1, 1, 3, 3);
    grad_out.setConstant(2.0f);
    Tensor grad_in = d.backward(grad_out);

    for (int i = 0; i < grad_out.size(); ++i)
        EXPECT_NEAR(grad_in[i], grad_out[i], EPS);
}

TEST(DropoutBackward, TrainMode_MaskApplied_ZerosAligned) {
    DropoutLayer d(0.5f);
    d.train();

    Tensor input(1, 1, 8, 8);
    input.setConstant(1.0f);
    Tensor out = d.forward(input); // génère le masque

    Tensor grad_out(1, 1, 8, 8);
    grad_out.setConstant(1.0f);
    Tensor grad_in = d.backward(grad_out);

    // Les zéros de out (unités droputées) doivent correspondre
    // aux zéros du gradient
    for (int i = 0; i < out.size(); ++i) {
        if (out[i] == 0.0f) {
            EXPECT_FLOAT_EQ(grad_in[i], 0.0f);
        } else {
            EXPECT_GT(grad_in[i], 0.0f);
        }
    }
}

TEST(DropoutBackward, TrainMode_ShapePreserved) {
    DropoutLayer d(0.4f);
    d.train();
    Tensor input(2, 3, 4, 4);
    input.setRandom();
    d.forward(input);
    Tensor grad_out(input.shape());
    grad_out.setConstant(1.0f);
    Tensor grad_in = d.backward(grad_out);
    EXPECT_EQ(grad_in.size(), input.size());
}

TEST(DropoutBackward, WithoutForward_Throws) {
    DropoutLayer d(0.5f);
    d.train();
    // backward sans avoir appelé forward → masque vide → doit lancer
    Tensor grad_out(1, 1, 2, 2);
    grad_out.setConstant(1.0f);
    EXPECT_THROW(d.backward(grad_out), std::runtime_error);
}

// =============================================================================
// ── setTraining / train / eval ───────────────────────────────────────────────
// =============================================================================

TEST(DropoutMode, SetTrainingTrue_SomeDropout) {
    DropoutLayer d(0.5f);
    d.setTraining(true);
    Tensor input(1, 1, 50, 50);
    input.setConstant(1.0f);
    Tensor out = d.forward(input);
    bool has_zero = false;
    for (int i = 0; i < out.size(); ++i)
        if (out[i] == 0.0f) { has_zero = true; break; }
    EXPECT_TRUE(has_zero);
}

TEST(DropoutMode, SetTrainingFalse_NoDropout) {
    DropoutLayer d(0.5f);
    d.setTraining(false);
    Tensor input(1, 1, 8, 8);
    input.setConstant(3.0f);
    Tensor out = d.forward(input);
    for (int i = 0; i < out.size(); ++i)
        EXPECT_NEAR(out[i], 3.0f, EPS);
}
