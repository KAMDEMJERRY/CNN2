// tests/test_ActivationLayer.cpp

#include <gtest/gtest.h>
#include "ActivationLayer.hpp"
#include <cmath>

static constexpr float EPS = 1e-5f;

// =============================================================================
// ── ReLULayer ─────────────────────────────────────────────────────────────────
// =============================================================================

TEST(ReLU, ForwardPositiveUnchanged) {
    ReLULayer relu;
    Tensor input(1, 1, 2, 2);
    input[0] = 1.0f; input[1] = 2.0f; input[2] = 0.5f; input[3] = 10.0f;
    Tensor out = relu.forward(input);
    for (int i = 0; i < out.size(); ++i)
        EXPECT_FLOAT_EQ(out[i], input[i]);
}

TEST(ReLU, ForwardNegativeZeroed) {
    ReLULayer relu;
    Tensor input(1, 1, 1, 4);
    input[0] = -1.0f; input[1] = -0.5f; input[2] = 0.0f; input[3] = -100.0f;
    Tensor out = relu.forward(input);
    for (int i = 0; i < out.size(); ++i)
        EXPECT_FLOAT_EQ(out[i], 0.0f);
}

TEST(ReLU, ForwardMixed) {
    ReLULayer relu;
    Tensor input(1, 1, 1, 4);
    input[0] = -3.0f; input[1] = 0.0f; input[2] = 2.0f; input[3] = -1.0f;
    Tensor out = relu.forward(input);
    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);
    EXPECT_NEAR(out[2],  2.0f, EPS);
    EXPECT_FLOAT_EQ(out[3], 0.0f);
}

TEST(ReLU, ForwardShapePreserved) {
    ReLULayer relu;
    Tensor input(2, 3, 4, 5);
    input.setRandom();
    Tensor out = relu.forward(input);
    EXPECT_EQ(out.dim(0), 2);
    EXPECT_EQ(out.dim(1), 3);
    EXPECT_EQ(out.dim(2), 4);
    EXPECT_EQ(out.dim(3), 5);
}

TEST(ReLU, BackwardMask) {
    ReLULayer relu;
    Tensor input(1, 1, 1, 3);
    input[0] = -1.0f; input[1] = 2.0f; input[2] = -0.5f;
    relu.forward(input);

    Tensor grad_out(1, 1, 1, 3);
    grad_out[0] = 5.0f; grad_out[1] = 5.0f; grad_out[2] = 5.0f;
    Tensor grad_in = relu.backward(grad_out);

    EXPECT_FLOAT_EQ(grad_in[0], 0.0f); // input[0] <= 0
    EXPECT_NEAR(grad_in[1], 5.0f, EPS); // input[1] > 0
    EXPECT_FLOAT_EQ(grad_in[2], 0.0f); // input[2] <= 0
}

TEST(ReLU, BackwardShapePreserved) {
    ReLULayer relu;
    Tensor input(2, 4, 3, 3);
    input.setRandom();
    Tensor out = relu.forward(input);
    Tensor grad_out(out.shape());
    grad_out.setConstant(1.0f);
    Tensor grad_in = relu.backward(grad_out);
    EXPECT_EQ(grad_in.size(), input.size());
}

TEST(ReLU, GetName) {
    ReLULayer relu;
    EXPECT_EQ(relu.getName(), "ReLU");
}

// =============================================================================
// ── LeakyReLULayer ───────────────────────────────────────────────────────────
// =============================================================================

TEST(LeakyReLU, ForwardPositiveUnchanged) {
    LeakyReLULayer lrelu(0.1f);
    Tensor input(1, 1, 1, 3);
    input[0] = 1.0f; input[1] = 2.5f; input[2] = 0.5f;
    Tensor out = lrelu.forward(input);
    for (int i = 0; i < out.size(); ++i)
        EXPECT_NEAR(out[i], input[i], EPS);
}

TEST(LeakyReLU, ForwardNegativeScaledByAlpha) {
    const float alpha = 0.2f;
    LeakyReLULayer lrelu(alpha);
    Tensor input(1, 1, 1, 3);
    input[0] = -2.0f; input[1] = -1.0f; input[2] = -0.5f;
    Tensor out = lrelu.forward(input);
    for (int i = 0; i < out.size(); ++i)
        EXPECT_NEAR(out[i], alpha * input[i], EPS);
}

TEST(LeakyReLU, BackwardPositiveGrad) {
    const float alpha = 0.1f;
    LeakyReLULayer lrelu(alpha);
    Tensor input(1, 1, 1, 2);
    input[0] = 1.0f; input[1] = -1.0f;
    lrelu.forward(input);

    Tensor grad_out(1, 1, 1, 2);
    grad_out[0] = 1.0f; grad_out[1] = 1.0f;
    Tensor grad_in = lrelu.backward(grad_out);

    EXPECT_NEAR(grad_in[0], 1.0f,  EPS); // positive → grad intact
    EXPECT_NEAR(grad_in[1], alpha, EPS); // negative → alpha * grad
}

TEST(LeakyReLU, GetName) {
    LeakyReLULayer lrelu;
    EXPECT_EQ(lrelu.getName(), "LeakyReLU");
}

// =============================================================================
// ── SigmoidLayer ─────────────────────────────────────────────────────────────
// =============================================================================

TEST(Sigmoid, ForwardRangeInZeroOne) {
    SigmoidLayer sig;
    Tensor input(1, 1, 1, 4);
    input[0] = -100.0f; input[1] = -1.0f; input[2] = 1.0f; input[3] = 100.0f;
    Tensor out = sig.forward(input);
    for (int i = 0; i < out.size(); ++i) {
        EXPECT_GE(out[i], 0.0f);
        EXPECT_LE(out[i], 1.0f);
    }
}

TEST(Sigmoid, ForwardZeroInput_HalfOutput) {
    SigmoidLayer sig;
    Tensor input(1, 1, 1, 1);
    input[0] = 0.0f;
    Tensor out = sig.forward(input);
    EXPECT_NEAR(out[0], 0.5f, EPS);
}

TEST(Sigmoid, BackwardShapePreserved) {
    SigmoidLayer sig;
    Tensor input(2, 3, 4, 4);
    input.setRandom();
    Tensor out = sig.forward(input);
    Tensor grad_out(out.shape());
    grad_out.setConstant(1.0f);
    Tensor grad_in = sig.backward(grad_out);
    EXPECT_EQ(grad_in.size(), input.size());
}

TEST(Sigmoid, BackwardGradMax_AtZero) {
    // Gradient σ'(0) = σ(0)(1-σ(0)) = 0.5 * 0.5 = 0.25
    SigmoidLayer sig;
    Tensor input(1, 1, 1, 1);
    input[0] = 0.0f;
    sig.forward(input);
    Tensor grad_out(1, 1, 1, 1);
    grad_out[0] = 1.0f;
    Tensor grad_in = sig.backward(grad_out);
    EXPECT_NEAR(grad_in[0], 0.25f, EPS);
}

TEST(Sigmoid, BackwardGradNearZero_LaargePosInput) {
    // Pour x très grand, σ'(x) ≈ 0
    SigmoidLayer sig;
    Tensor input(1, 1, 1, 1);
    input[0] = 30.0f;
    sig.forward(input);
    Tensor grad_out(1, 1, 1, 1);
    grad_out[0] = 1.0f;
    Tensor grad_in = sig.backward(grad_out);
    EXPECT_NEAR(grad_in[0], 0.0f, 1e-3f);
}

TEST(Sigmoid, GetName) {
    SigmoidLayer sig;
    EXPECT_EQ(sig.getName(), "Sigmoid");
}

// =============================================================================
// ── SoftmaxLayer ─────────────────────────────────────────────────────────────
// =============================================================================

TEST(Softmax, ForwardSumToOne) {
    SoftmaxLayer softmax;
    // (B=2, C=4, H=1, W=1)
    Tensor input(2, 4, 1, 1);
    input.setRandom();
    Tensor out = softmax.forward(input);

    for (int b = 0; b < 2; ++b) {
        float sum = 0.0f;
        for (int c = 0; c < 4; ++c) sum += out(b, c, 0, 0);
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }
}

TEST(Softmax, ForwardOutputPositive) {
    SoftmaxLayer softmax;
    Tensor input(1, 3, 1, 1);
    input.setRandom();
    Tensor out = softmax.forward(input);
    for (int i = 0; i < out.size(); ++i)
        EXPECT_GT(out[i], 0.0f);
}

TEST(Softmax, ForwardNumericalStability) {
    // Avec des logits très grands, le softmax ne doit pas produire NaN/Inf
    SoftmaxLayer softmax;
    Tensor input(1, 3, 1, 1);
    input[0] = 1000.0f; input[1] = 1001.0f; input[2] = 1002.0f;
    Tensor out = softmax.forward(input);
    for (int i = 0; i < out.size(); ++i) {
        EXPECT_FALSE(std::isnan(out[i]));
        EXPECT_FALSE(std::isinf(out[i]));
    }
}

TEST(Softmax, ForwardShapePreserved) {
    SoftmaxLayer softmax;
    Tensor input(3, 5, 1, 1);
    input.setRandom();
    Tensor out = softmax.forward(input);
    EXPECT_EQ(out.dim(0), 3);
    EXPECT_EQ(out.dim(1), 5);
}

TEST(Softmax, ForwardInvalidSpatialDims_Throws) {
    SoftmaxLayer softmax;
    Tensor input(1, 4, 2, 2); // H et W != 1
    input.setRandom();
    EXPECT_THROW(softmax.forward(input), std::runtime_error);
}

TEST(Softmax, BackwardShapePreserved) {
    SoftmaxLayer softmax;
    Tensor input(2, 4, 1, 1);
    input.setRandom();
    Tensor out = softmax.forward(input);
    Tensor grad_out(out.shape());
    grad_out.setConstant(0.1f);
    Tensor grad_in = softmax.backward(grad_out);
    EXPECT_EQ(grad_in.size(), input.size());
}

TEST(Softmax, GetPredictions_ArgMax) {
    SoftmaxLayer softmax;
    // Logit très haut pour classe 2 → prédiction = 2
    Tensor input(1, 4, 1, 1);
    input.setZero();
    input[2] = 100.0f; // classe 2 domine
    softmax.forward(input);
    Tensor preds = softmax.getPredictions();
    EXPECT_FLOAT_EQ(preds[0], 2.0f);
}

TEST(Softmax, GetName) {
    SoftmaxLayer softmax;
    EXPECT_EQ(softmax.getName(), "Softmax");
}
