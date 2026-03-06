// tests/test_Dimensions.cpp

#include <gtest/gtest.h>
#include "Dimensions.hpp"

// =============================================================================
// ── DimensionCalculator ──────────────────────────────────────────────────────
// =============================================================================

TEST(DimCalcConv, NoPadding_Stride1) {
    DimensionCalculator::ConvParams p;
    p.in_channels = 1;
    p.out_channels = 4;
    p.kernel_h = 3;
    p.kernel_w = 3;
    p.stride_h = 1;
    p.stride_w = 1;
    p.pad_h = 0;
    p.pad_w = 0;

    auto [h, w, c] = DimensionCalculator::convOutput(8, 8, 1, p);
    EXPECT_EQ(h, 6); // (8 - 3)/1 + 1 = 6
    EXPECT_EQ(w, 6);
    EXPECT_EQ(c, 4);
}

TEST(DimCalcConv, Padding_Stride2) {
    DimensionCalculator::ConvParams p;
    p.out_channels = 8;
    p.kernel_h = 3;
    p.kernel_w = 3;
    p.stride_h = 2;
    p.stride_w = 2;
    p.pad_h = 1;
    p.pad_w = 1;

    auto [h, w, c] = DimensionCalculator::convOutput(7, 7, 3, p);
    // (7 + 2*1 - 3)/2 + 1 = 6/2 + 1 = 4
    EXPECT_EQ(h, 4);
    EXPECT_EQ(w, 4);
    EXPECT_EQ(c, 8);
}

TEST(DimCalcPool, DefaultParams) {
    DimensionCalculator::PoolParams p;
    p.pool_size = 2;
    p.stride = 2;

    auto [h, w, c] = DimensionCalculator::poolOutput(8, 8, 5, p);
    EXPECT_EQ(h, 4); // (8 - 2)/2 + 1 = 4
    EXPECT_EQ(w, 4);
    EXPECT_EQ(c, 5);
}

TEST(DimCalcFlatten, BasicCalculation) {
    int flat = DimensionCalculator::flattenSize(4, 5, 3); // H=4, W=5, C=3
    EXPECT_EQ(flat, 60);
}
