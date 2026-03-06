// tests/test_Tensor.cpp
//
// Compilation standalone :
//   g++ -std=c++20 -O2 test_Tensor.cpp \
//       -I.. -I/usr/include/eigen3 \
//       -lgtest -lgtest_main -lpthread -o test_Tensor && ./test_Tensor

#include <gtest/gtest.h>
#include "Tensor.hpp"
#include <cmath>
#include <vector>

// =============================================================================
// Helpers locaux
// =============================================================================

static constexpr float EPS = 1e-5f;

// Vérifie que tous les éléments valent `expected`
static void expectAllEq(const Tensor& t, float expected) {
    for (int i = 0; i < t.size(); ++i)
        EXPECT_FLOAT_EQ(t[i], expected) << "index " << i;
}

// =============================================================================
// ── Construction ─────────────────────────────────────────────────────────────
// =============================================================================

TEST(TensorConstruction, Default_IsRank4_Empty) {
    Tensor t;
    EXPECT_EQ(t.ndim(), 4);
    EXPECT_EQ(t.size(), 0);
}

TEST(TensorConstruction, FourArgs_IsRank4) {
    Tensor t(2, 3, 4, 5);
    EXPECT_EQ(t.ndim(), 4);
}

TEST(TensorConstruction, FourArgs_DimsCorrect) {
    Tensor t(2, 3, 4, 5);
    EXPECT_EQ(t.dim(0), 2);  // B
    EXPECT_EQ(t.dim(1), 3);  // C
    EXPECT_EQ(t.dim(2), 4);  // H
    EXPECT_EQ(t.dim(3), 5);  // W
}

TEST(TensorConstruction, FourArgs_SizeCorrect) {
    Tensor t(2, 3, 4, 5);
    EXPECT_EQ(t.size(), 2 * 3 * 4 * 5);
}

TEST(TensorConstruction, FiveArgs_IsRank5) {
    Tensor t(2, 3, 4, 5, 6);
    EXPECT_EQ(t.ndim(), 5);
}

TEST(TensorConstruction, FiveArgs_DimsCorrect) {
    Tensor t(2, 3, 4, 5, 6);
    EXPECT_EQ(t.dim(0), 2);  // B
    EXPECT_EQ(t.dim(1), 3);  // C
    EXPECT_EQ(t.dim(2), 4);  // D
    EXPECT_EQ(t.dim(3), 5);  // H
    EXPECT_EQ(t.dim(4), 6);  // W
}

TEST(TensorConstruction, FiveArgs_SizeCorrect) {
    Tensor t(2, 3, 4, 5, 6);
    EXPECT_EQ(t.size(), 2 * 3 * 4 * 5 * 6);
}

TEST(TensorConstruction, FromShape4D) {
    Tensor t(std::vector<int>{2, 3, 4, 5});
    EXPECT_EQ(t.ndim(), 4);
    EXPECT_EQ(t.dim(0), 2);
    EXPECT_EQ(t.size(), 2 * 3 * 4 * 5);
}

TEST(TensorConstruction, FromShape5D) {
    Tensor t(std::vector<int>{1, 2, 3, 4, 5});
    EXPECT_EQ(t.ndim(), 5);
    EXPECT_EQ(t.dim(2), 3);
}

TEST(TensorConstruction, FromShapeInvalidRankThrows) {
    EXPECT_THROW(Tensor(std::vector<int>{2, 3}),       std::runtime_error);
    EXPECT_THROW(Tensor(std::vector<int>{1, 2, 3}),    std::runtime_error);
    EXPECT_THROW(Tensor(std::vector<int>{1,2,3,4,5,6}),std::runtime_error);
}

// =============================================================================
// ── Copie / déplacement ──────────────────────────────────────────────────────
// =============================================================================

TEST(TensorCopy, CopyConstructor_IndependentData) {
    Tensor a(1, 1, 3, 3);
    a.setConstant(5.0f);
    Tensor b = a;
    b.setConstant(0.0f);
    // a doit rester inchangé
    expectAllEq(a, 5.0f);
}

TEST(TensorCopy, CopyAssignment_IndependentData) {
    Tensor a(1, 1, 3, 3);
    a.setConstant(7.0f);
    Tensor b(1, 1, 3, 3);
    b = a;
    a.setConstant(0.0f);
    expectAllEq(b, 7.0f);
}

TEST(TensorCopy, CopyPreservesRank) {
    Tensor a(1, 2, 3, 4, 5);
    Tensor b = a;
    EXPECT_EQ(b.ndim(), 5);
}

TEST(TensorCopy, MoveConstructor_LeavesSourceEmpty) {
    Tensor a(1, 1, 4, 4);
    a.setConstant(1.0f);
    Tensor b = std::move(a);
    EXPECT_EQ(b.size(), 16);
}

// =============================================================================
// ── Initialisation ───────────────────────────────────────────────────────────
// =============================================================================

TEST(TensorInit, SetZero_AllZeros) {
    Tensor t(2, 3, 4, 4);
    t.setConstant(99.0f);
    t.setZero();
    expectAllEq(t, 0.0f);
}

TEST(TensorInit, SetConstant) {
    Tensor t(1, 2, 3, 3);
    t.setConstant(42.0f);
    expectAllEq(t, 42.0f);
}

TEST(TensorInit, SetRandom_NotAllZero) {
    Tensor t(2, 2, 4, 4);
    t.setZero();
    t.setRandom();
    float norm = 0.f;
    for (int i = 0; i < t.size(); ++i) norm += t[i] * t[i];
    EXPECT_GT(norm, 0.f);
}

// =============================================================================
// ── Accès aux éléments ───────────────────────────────────────────────────────
// =============================================================================

TEST(TensorAccess, FlatIndex_ReadWrite) {
    Tensor t(1, 1, 2, 2);
    t.setZero();
    t[3] = 99.0f;
    EXPECT_FLOAT_EQ(t[3], 99.0f);
}

TEST(TensorAccess, 4D_Operator_ReadWrite) {
    Tensor t(1, 2, 3, 4);
    t.setZero();
    t(0, 1, 2, 3) = 13.0f;
    EXPECT_FLOAT_EQ(t(0, 1, 2, 3), 13.0f);
}

TEST(TensorAccess, 5D_Operator_ReadWrite) {
    Tensor t(1, 2, 3, 4, 5);
    t.setZero();
    t(0, 1, 2, 3, 4) = 77.0f;
    EXPECT_FLOAT_EQ(t(0, 1, 2, 3, 4), 77.0f);
}

TEST(TensorAccess, 4D_And_5D_SameMemory_Rank5_D1) {
    // Un tenseur 5D avec D=1 doit pouvoir être lu via operator(b,c,0,h,w)
    Tensor t(1, 1, 1, 3, 3);
    t.setZero();
    t(0, 0, 0, 1, 2) = 55.0f;
    EXPECT_FLOAT_EQ(t(0, 0, 0, 1, 2), 55.0f);
}

TEST(TensorAccess, DimOutOfRange_4D_Throws) {
    Tensor t(1, 2, 3, 4);
    EXPECT_THROW(t.dim(4), std::out_of_range);
    EXPECT_THROW(t.dim(-1), std::out_of_range);
}

TEST(TensorAccess, DimOutOfRange_5D_Throws) {
    Tensor t(1, 2, 3, 4, 5);
    EXPECT_THROW(t.dim(5), std::out_of_range);
}

// =============================================================================
// ── shape() ──────────────────────────────────────────────────────────────────
// =============================================================================

TEST(TensorShape, Shape4D_ReturnsCorrectVector) {
    Tensor t(2, 3, 4, 5);
    auto s = t.shape();
    ASSERT_EQ(s.size(), 4u);
    EXPECT_EQ(s[0], 2); EXPECT_EQ(s[1], 3);
    EXPECT_EQ(s[2], 4); EXPECT_EQ(s[3], 5);
}

TEST(TensorShape, Shape5D_ReturnsCorrectVector) {
    Tensor t(2, 3, 4, 5, 6);
    auto s = t.shape();
    ASSERT_EQ(s.size(), 5u);
    EXPECT_EQ(s[0], 2); EXPECT_EQ(s[4], 6);
}

TEST(TensorShape, Shape4D_DoesNotExposeInternalD1) {
    // Le mode 4D stocke D=1 en interne mais shape() doit rester [B,C,H,W]
    Tensor t(2, 3, 4, 5);
    EXPECT_EQ(t.shape().size(), 4u);
}

// =============================================================================
// ── Conversion de rang : as4D / as5D ─────────────────────────────────────────
// =============================================================================

TEST(TensorConversion, As5D_From4D_IsRank5) {
    Tensor t4(1, 2, 3, 4);
    Tensor t5 = t4.as5D();
    EXPECT_EQ(t5.ndim(), 5);
}

TEST(TensorConversion, As5D_PreservesSize) {
    Tensor t4(1, 2, 3, 4);
    EXPECT_EQ(t4.as5D().size(), t4.size());
}

TEST(TensorConversion, As5D_IdempotentOnRank5) {
    Tensor t5(1, 2, 3, 4, 5);
    EXPECT_EQ(t5.as5D().ndim(), 5);
}

TEST(TensorConversion, As4D_From5D_D1_IsRank4) {
    Tensor t5(1, 2, 1, 4, 5); // D=1
    Tensor t4 = t5.as4D();
    EXPECT_EQ(t4.ndim(), 4);
}

TEST(TensorConversion, As4D_From5D_D1_PreservesData) {
    Tensor t5(1, 1, 1, 3, 3);
    t5.setZero();
    t5(0, 0, 0, 1, 2) = 42.0f;
    Tensor t4 = t5.as4D();
    EXPECT_NEAR(t4(0, 0, 1, 2), 42.0f, EPS);
}

TEST(TensorConversion, As4D_IdempotentOnRank4) {
    Tensor t4(1, 2, 3, 4);
    EXPECT_EQ(t4.as4D().ndim(), 4);
}

TEST(TensorConversion, As4D_ThrowsWhenDepthNot1) {
    Tensor t5(1, 2, 3, 4, 5); // D=3
    EXPECT_THROW(t5.as4D(), std::runtime_error);
}

// =============================================================================
// ── Reshape ──────────────────────────────────────────────────────────────────
// =============================================================================

TEST(TensorReshape, PreservesSize) {
    Tensor t(1, 6, 1, 1);
    t.setConstant(1.0f);
    Tensor r = t.reshape({1, 2, 3, 1});
    EXPECT_EQ(r.size(), t.size());
}

TEST(TensorReshape, PreservesData) {
    Tensor t(1, 1, 2, 3);
    for (int i = 0; i < t.size(); ++i) t[i] = static_cast<float>(i);
    Tensor r = t.reshape({1, 1, 3, 2});
    for (int i = 0; i < t.size(); ++i)
        EXPECT_FLOAT_EQ(r[i], static_cast<float>(i));
}

TEST(TensorReshape, IncompatibleSizeThrows) {
    Tensor t(1, 1, 3, 3); // size = 9
    EXPECT_THROW(t.reshape({1, 1, 4, 4}), std::runtime_error); // size = 16
}

// =============================================================================
// ── Eigen interop ────────────────────────────────────────────────────────────
// =============================================================================

TEST(TensorEigen, ToMatrix_CorrectDimensions) {
    Tensor t(4, 3, 1, 1); // B=4, flat=3
    t.setRandom();
    Eigen::MatrixXf m = t.toMatrix();
    EXPECT_EQ(m.rows(), 4);
    EXPECT_EQ(m.cols(), 3);
}

TEST(TensorEigen, FromMatrix_RoundTrip4D) {
    Tensor orig(2, 3, 1, 1);
    for (int i = 0; i < orig.size(); ++i) orig[i] = static_cast<float>(i);
    Eigen::MatrixXf m = orig.toMatrix();
    Tensor back = Tensor::fromMatrix(m, 2, 3, 1, 1);
    for (int i = 0; i < orig.size(); ++i)
        EXPECT_NEAR(back[i], orig[i], EPS);
}

TEST(TensorEigen, GetDataPointer_NotNull) {
    Tensor t(1, 1, 2, 2);
    EXPECT_NE(t.getData(), nullptr);
}

// =============================================================================
// ── dim5 (accès interne brut) ────────────────────────────────────────────────
// =============================================================================

TEST(TensorDim5, Rank4_InternalDepthIs1) {
    Tensor t(2, 3, 4, 5);
    // stocké comme (2,3,1,4,5)
    EXPECT_EQ(t.dim5(2), 1);
}

TEST(TensorDim5, Rank5_MatchesDim) {
    Tensor t(2, 3, 4, 5, 6);
    EXPECT_EQ(t.dim5(0), 2);
    EXPECT_EQ(t.dim5(2), 4);
    EXPECT_EQ(t.dim5(4), 6);
}

TEST(TensorDim5, OutOfRangeThrows) {
    Tensor t(1, 1, 1, 1);
    EXPECT_THROW(t.dim5(5),  std::out_of_range);
    EXPECT_THROW(t.dim5(-1), std::out_of_range);
}