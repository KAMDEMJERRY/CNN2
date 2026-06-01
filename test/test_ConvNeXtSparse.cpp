#include <gtest/gtest.h>
#include "Tensor.hpp"
#include "SparseTensor.hpp"
#include "PatchifyStem3DSparse.hpp"
#include "SparseDownsample3D.hpp"
#include "ConvNeXtBlock3DSparse.hpp"
#include "LayerNorm3DSparse.hpp"

// -----------------------------------------------------------------------------
// Test PatchifyStem3DSparse
// -----------------------------------------------------------------------------
TEST(ConvNeXtSparseTest, PatchifyStem3DShape) {
    int B = 1, C = 1, D = 28, H = 28, W = 28;
    Tensor input(B, C, D, H, W);
    input.setZero();
    input(0, 0, 10, 10, 10) = 1.0f; // Active voxel
    input(0, 0, 20, 20, 20) = 1.0f;

    PatchifyStem3DSparse stem(C, 32, 4, 0.0f);
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
// Test SparseDownsample3D
// -----------------------------------------------------------------------------
TEST(ConvNeXtSparseTest, SparseDownsample3DShape) {
    int B = 1, C = 32, D = 7, H = 7, W = 7;
    Tensor input(B, C, D, H, W);
    input.setZero();
    input(0, 0, 3, 3, 3) = 1.0f;

    SparseDownsample3D down(C, 64, 0.0f);
    Tensor out = down.forward(input);

    EXPECT_EQ(out.dim(0), B);
    EXPECT_EQ(out.dim(1), 64);
    EXPECT_EQ(out.dim(2), 3);
    EXPECT_EQ(out.dim(3), 3);
    EXPECT_EQ(out.dim(4), 3);
}

// -----------------------------------------------------------------------------
// Test ConvNeXtBlock3DSparse
// -----------------------------------------------------------------------------
TEST(ConvNeXtSparseTest, ConvNeXtBlock3DSparseShape) {
    int B = 2, C = 16, D = 7, H = 7, W = 7;
    Tensor input(B, C, D, H, W);
    input.setZero();
    input(0, 0, 1, 1, 1) = 2.0f;

    ConvNeXtBlock3DSparse block(C, 7, 1e-6f, 0.0f);
    Tensor out = block.forward(input);

    EXPECT_EQ(out.shape(), input.shape());

    Tensor grad(out.shape());
    grad.setZero();
    Tensor gradIn = block.backward(grad);
    EXPECT_EQ(gradIn.shape(), input.shape());
}
