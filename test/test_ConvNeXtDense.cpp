#include <gtest/gtest.h>
#include "Tensor.hpp"
#include "DepthwiseConvLayer3D.hpp"
#include "LayerNorm3DLayer.hpp"
#include "PatchifyStem3D.hpp"
#include "DenseDownsample3D.hpp"
#include "ConvNeXtBlock3D.hpp"

// -----------------------------------------------------------------------------
// Test DepthwiseConvLayer3D
// -----------------------------------------------------------------------------
TEST(ConvNeXtDenseTest, DepthwiseConv3DShape) {
    int B = 2, C = 3, D = 5, H = 5, W = 5;
    Tensor input(B, C, D, H, W);
    input.setZero();

    DepthwiseConvLayer3D dw(C, 3, 3, 3, 1, 1, 1, 1, 1, 1); // k=3, pad=1 => shape preserved
    Tensor out = dw.forward(input);

    EXPECT_EQ(out.dim(0), B);
    EXPECT_EQ(out.dim(1), C);
    EXPECT_EQ(out.dim(2), D);
    EXPECT_EQ(out.dim(3), H);
    EXPECT_EQ(out.dim(4), W);

    // backward
    Tensor grad(out.shape());
    grad.setZero();
    Tensor gradIn = dw.backward(grad);
    EXPECT_EQ(gradIn.shape(), input.shape());
}

// -----------------------------------------------------------------------------
// Test LayerNorm3DLayer
// -----------------------------------------------------------------------------
TEST(ConvNeXtDenseTest, LayerNorm3DShapeAndValues) {
    int B = 1, C = 4, D = 2, H = 2, W = 2;
    Tensor input(B, C, D, H, W);
    // Fill with values
    for (int c = 0; c < C; ++c) {
        for (int d = 0; d < D; ++d) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    input(0, c, d, h, w) = (c + 1) * 1.0f; // channels 1, 2, 3, 4
                }
            }
        }
    }

    LayerNorm3DLayer ln(C);
    Tensor out = ln.forward(input);

    // After LN, mean across dimension C should be 0, std 1
    // for each voxel. Input was [1, 2, 3, 4]. Mean = 2.5, var = 1.25.
    // std = ~1.118 => (1-2.5)/1.118 = -1.34
    // Gamma = 1, Beta = 0.
    EXPECT_EQ(out.shape(), input.shape());
    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        sum += out(0, c, 1, 1, 1);
    }
    EXPECT_NEAR(sum, 0.0f, 1e-4);

    Tensor grad(out.shape());
    grad.setZero();
    Tensor gradIn = ln.backward(grad);
    EXPECT_EQ(gradIn.shape(), input.shape());
}

// -----------------------------------------------------------------------------
// Test PatchifyStem3D
// -----------------------------------------------------------------------------
TEST(ConvNeXtDenseTest, PatchifyStem3DShape) {
    int B = 1, C = 1, D = 28, H = 28, W = 28;
    Tensor input(B, C, D, H, W);
    input.setZero();

    PatchifyStem3D stem(C, 32, 4); // 1 -> 32, patch 4
    Tensor out = stem.forward(input);

    EXPECT_EQ(out.dim(0), B);
    EXPECT_EQ(out.dim(1), 32);
    EXPECT_EQ(out.dim(2), 7);
    EXPECT_EQ(out.dim(3), 7);
    EXPECT_EQ(out.dim(4), 7);

    Tensor grad(out.shape());
    grad.setZero();
    Tensor gradIn = stem.backward(grad);
    EXPECT_EQ(gradIn.shape(), input.shape());
}

// -----------------------------------------------------------------------------
// Test DenseDownsample3D
// -----------------------------------------------------------------------------
TEST(ConvNeXtDenseTest, DenseDownsample3DShape) {
    int B = 1, C = 32, D = 7, H = 7, W = 7; // downsample 7 to 3
    Tensor input(B, C, D, H, W);
    input.setZero();

    DenseDownsample3D down(C, 64);
    Tensor out = down.forward(input);

    EXPECT_EQ(out.dim(0), B);
    EXPECT_EQ(out.dim(1), 64);
    // (7 - 2)/2 + 1 = 3
    EXPECT_EQ(out.dim(2), 3);
    EXPECT_EQ(out.dim(3), 3);
    EXPECT_EQ(out.dim(4), 3);
}

// -----------------------------------------------------------------------------
// Test ConvNeXtBlock3D
// -----------------------------------------------------------------------------
TEST(ConvNeXtDenseTest, ConvNeXtBlock3DShape) {
    int B = 2, C = 16, D = 7, H = 7, W = 7;
    Tensor input(B, C, D, H, W);
    input.setZero();

    ConvNeXtBlock3D block(C, 7); // k=7
    Tensor out = block.forward(input);

    EXPECT_EQ(out.shape(), input.shape());

    Tensor grad(out.shape());
    grad.setZero();
    Tensor gradIn = block.backward(grad);
    EXPECT_EQ(gradIn.shape(), input.shape());
}
