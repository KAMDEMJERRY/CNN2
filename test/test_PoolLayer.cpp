// tests/test_PoolLayer.cpp

#include <gtest/gtest.h>
#include "PoolLayer.hpp"
#include <cmath>

static constexpr float EPS = 1e-5f;

// =============================================================================
// ── MaxPoolLayer (2D) ─────────────────────────────────────────────────────────
// =============================================================================

TEST(MaxPool2D, OutputShape_Default) {
    // (B=1, C=2, H=4, W=4), pool=2, stride=2 → out (1,2,2,2)
    MaxPoolLayer pool(2, 2);
    Tensor input(1, 2, 4, 4);
    input.setRandom();
    Tensor out = pool.forward(input);
    EXPECT_EQ(out.dim(0), 1);
    EXPECT_EQ(out.dim(1), 2);
    EXPECT_EQ(out.dim(2), 2);
    EXPECT_EQ(out.dim(3), 2);
}

TEST(MaxPool2D, OutputShape_BatchPreserved) {
    MaxPoolLayer pool(2, 2);
    for (int B : {1, 4, 8}) {
        Tensor input(B, 3, 8, 8);
        input.setRandom();
        EXPECT_EQ(pool.forward(input).dim(0), B);
    }
}

TEST(MaxPool2D, OutputShape_Rectangular) {
    // pool (2, 4), stride (1, 2) sur (1, 1, 4, 8)
    MaxPoolLayer pool(2, 4, 1, 2);
    Tensor input(1, 1, 4, 8);
    input.setRandom();
    Tensor out = pool.forward(input);
    // H: (4-2)/1+1 = 3, W: (8-4)/2+1 = 3
    EXPECT_EQ(out.dim(2), 3);
    EXPECT_EQ(out.dim(3), 3);
}

TEST(MaxPool2D, ForwardSelectsMax) {
    MaxPoolLayer pool(2, 2);
    Tensor input(1, 1, 2, 2);
    input.setZero();
    input(0, 0, 0, 0) = 1.0f;
    input(0, 0, 0, 1) = 2.0f;
    input(0, 0, 1, 0) = 3.0f;
    input(0, 0, 1, 1) = 4.0f;     // max = 4
    Tensor out = pool.forward(input);
    EXPECT_NEAR(out(0, 0, 0, 0), 4.0f, EPS);
}

TEST(MaxPool2D, ForwardThrows_WrongRank) {
    MaxPoolLayer pool(2, 2);
    Tensor input(1, 1, 4, 4, 4); // 5D
    EXPECT_THROW(pool.forward(input), std::runtime_error);
}

TEST(MaxPool2D, BackwardShape) {
    MaxPoolLayer pool(2, 2);
    Tensor input(2, 3, 8, 8);
    input.setRandom();
    Tensor out = pool.forward(input);

    Tensor grad_out(out.shape());
    grad_out.setConstant(1.0f);
    Tensor grad_in = pool.backward(grad_out);

    EXPECT_EQ(grad_in.dim(0), 2);
    EXPECT_EQ(grad_in.dim(1), 3);
    EXPECT_EQ(grad_in.dim(2), 8);
    EXPECT_EQ(grad_in.dim(3), 8);
}

TEST(MaxPool2D, BackwardGradRouted_ToMaxPosition) {
    // Avec input connu, le gradient doit aller au pixel max
    MaxPoolLayer pool(2, 2);
    Tensor input(1, 1, 2, 2);
    input.setZero();
    input(0, 0, 1, 1) = 5.0f; // max à (1,1)

    pool.forward(input);

    Tensor grad_out(1, 1, 1, 1);
    grad_out.setConstant(1.0f);
    Tensor grad_in = pool.backward(grad_out);

    EXPECT_NEAR(grad_in(0, 0, 1, 1), 1.0f, EPS); // gradient vers max
    EXPECT_NEAR(grad_in(0, 0, 0, 0), 0.0f, EPS); // zero ailleurs
    EXPECT_NEAR(grad_in(0, 0, 0, 1), 0.0f, EPS);
    EXPECT_NEAR(grad_in(0, 0, 1, 0), 0.0f, EPS);
}

TEST(MaxPool2D, GetName) {
    MaxPoolLayer pool;
    EXPECT_EQ(pool.getName(), "MaxPool2D");
}

// =============================================================================
// ── GlobalAvgPool2DLayer ──────────────────────────────────────────────────────
// =============================================================================

TEST(GlobalAvgPool2D, OutputShape) {
    GlobalAvgPool2DLayer gap;
    Tensor input(2, 4, 8, 8);
    input.setRandom();
    Tensor out = gap.forward(input);
    EXPECT_EQ(out.dim(0), 2);
    EXPECT_EQ(out.dim(1), 4);
    EXPECT_EQ(out.dim(2), 1);
    EXPECT_EQ(out.dim(3), 1);
}

TEST(GlobalAvgPool2D, ForwardComputesMean) {
    GlobalAvgPool2DLayer gap;
    Tensor input(1, 1, 2, 2);
    input.setZero();
    input(0, 0, 0, 0) = 1.0f;
    input(0, 0, 0, 1) = 2.0f;
    input(0, 0, 1, 0) = 3.0f;
    input(0, 0, 1, 1) = 4.0f;   // moyenne = 2.5
    Tensor out = gap.forward(input);
    EXPECT_NEAR(out(0, 0, 0, 0), 2.5f, EPS);
}

TEST(GlobalAvgPool2D, ForwardThrows_WrongRank) {
    GlobalAvgPool2DLayer gap;
    Tensor input(1, 1, 4, 4, 4); // 5D
    EXPECT_THROW(gap.forward(input), std::runtime_error);
}

TEST(GlobalAvgPool2D, BackwardShape) {
    GlobalAvgPool2DLayer gap;
    Tensor input(2, 3, 4, 4);
    input.setRandom();
    gap.forward(input);

    Tensor grad_out(2, 3, 1, 1);
    grad_out.setConstant(1.0f);
    Tensor grad_in = gap.backward(grad_out);

    EXPECT_EQ(grad_in.dim(2), 4);
    EXPECT_EQ(grad_in.dim(3), 4);
}

TEST(GlobalAvgPool2D, BackwardScale) {
    // Gradient doit être distribué uniformément
    GlobalAvgPool2DLayer gap;
    Tensor input(1, 1, 2, 2);
    input.setRandom();
    gap.forward(input);

    Tensor grad_out(1, 1, 1, 1);
    grad_out.setConstant(4.0f);
    Tensor grad_in = gap.backward(grad_out);

    // scale = 1/(H*W) = 0.25, grad = 4.0 * 0.25 = 1.0 pour chaque pixel
    for (int i = 0; i < grad_in.size(); ++i)
        EXPECT_NEAR(grad_in[i], 1.0f, EPS);
}

TEST(GlobalAvgPool2D, GetName) {
    GlobalAvgPool2DLayer gap;
    EXPECT_EQ(gap.getName(), "GlobalAvgPool2D");
}

// =============================================================================
// ── MaxPool3DLayer ────────────────────────────────────────────────────────────
// =============================================================================

TEST(MaxPool3D, OutputShape_Default) {
    // (B=1, C=2, D=4, H=4, W=4), pool=2, stride=2 → out (1,2,2,2,2)
    MaxPool3DLayer pool3d(2, 2);
    Tensor input(1, 2, 4, 4, 4);
    input.setRandom();
    Tensor out = pool3d.forward(input);
    EXPECT_EQ(out.dim(0), 1);
    EXPECT_EQ(out.dim(1), 2);
    EXPECT_EQ(out.dim(2), 2);
    EXPECT_EQ(out.dim(3), 2);
    EXPECT_EQ(out.dim(4), 2);
}

TEST(MaxPool3D, ForwardThrows_WrongRank) {
    MaxPool3DLayer pool3d(2, 2);
    Tensor input(1, 1, 4, 4); // 4D
    EXPECT_THROW(pool3d.forward(input), std::runtime_error);
}

TEST(MaxPool3D, ForwardSelectsMax) {
    MaxPool3DLayer pool3d(2, 2);
    Tensor input(1, 1, 2, 2, 2);
    input.setZero();
    input(0, 0, 1, 1, 1) = 9.0f; // max
    Tensor out = pool3d.forward(input);
    EXPECT_NEAR(out(0, 0, 0, 0, 0), 9.0f, EPS);
}

TEST(MaxPool3D, BackwardShape) {
    MaxPool3DLayer pool3d(2, 2);
    Tensor input(1, 2, 4, 4, 4);
    input.setRandom();
    Tensor out = pool3d.forward(input);

    Tensor grad_out(out.shape());
    grad_out.setConstant(1.0f);
    Tensor grad_in = pool3d.backward(grad_out);

    EXPECT_EQ(grad_in.dim(2), 4);
    EXPECT_EQ(grad_in.dim(3), 4);
    EXPECT_EQ(grad_in.dim(4), 4);
}

TEST(MaxPool3D, GetName) {
    MaxPool3DLayer pool3d;
    EXPECT_EQ(pool3d.getName(), "MaxPool3D");
}

// =============================================================================
// ── GlobalAvgPool3DLayer ──────────────────────────────────────────────────────
// =============================================================================

TEST(GlobalAvgPool3D, OutputShape) {
    GlobalAvgPool3DLayer gap3d;
    Tensor input(2, 4, 3, 4, 5);
    input.setRandom();
    Tensor out = gap3d.forward(input);
    EXPECT_EQ(out.dim(0), 2);
    EXPECT_EQ(out.dim(1), 4);
    EXPECT_EQ(out.dim(2), 1);
    EXPECT_EQ(out.dim(3), 1);
    EXPECT_EQ(out.dim(4), 1);
}

TEST(GlobalAvgPool3D, ForwardComputesMean) {
    GlobalAvgPool3DLayer gap3d;
    // Volume 1x1x8 rempli de 8.0 → moyenne = 8.0
    Tensor input(1, 1, 2, 2, 2);
    input.setConstant(8.0f);
    Tensor out = gap3d.forward(input);
    EXPECT_NEAR(out(0, 0, 0, 0, 0), 8.0f, EPS);
}

TEST(GlobalAvgPool3D, ForwardThrows_WrongRank) {
    GlobalAvgPool3DLayer gap3d;
    Tensor input(1, 1, 4, 4); // 4D
    EXPECT_THROW(gap3d.forward(input), std::runtime_error);
}

TEST(GlobalAvgPool3D, BackwardShape) {
    GlobalAvgPool3DLayer gap3d;
    Tensor input(1, 2, 3, 4, 5);
    input.setRandom();
    gap3d.forward(input);

    Tensor grad_out(1, 2, 1, 1, 1);
    grad_out.setConstant(1.0f);
    Tensor grad_in = gap3d.backward(grad_out);

    EXPECT_EQ(grad_in.dim(2), 3);
    EXPECT_EQ(grad_in.dim(3), 4);
    EXPECT_EQ(grad_in.dim(4), 5);
}

TEST(GlobalAvgPool3D, GetName) {
    GlobalAvgPool3DLayer gap3d;
    EXPECT_EQ(gap3d.getName(), "GlobalAvgPool3D");
}
