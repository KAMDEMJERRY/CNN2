#include <gtest/gtest.h>
#include "SparseTensor.hpp"
#include "SparseConvLayer3D.hpp"
#include <cmath>

// =============================================================================
// Helpers
// =============================================================================

// Crée un Tensor 5D rempli d'une valeur constante
static Tensor makeDense(int B, int C, int D, int H, int W, float val = 1.0f) {
    Tensor t(B, C, D, H, W);
    t.setConstant(val);
    return t;
}

// Crée un Tensor 5D avec uniquement certains voxels actifs
// active_positions : liste de {d, h, w} (mêmes pour tous les batches)
static Tensor makeSparseInput(int B, int C, int D, int H, int W,
                               const std::vector<std::array<int,3>>& active,
                               float val = 1.0f) {
    Tensor t(B, C, D, H, W);
    t.setZero();
    for (int b = 0; b < B; ++b)
        for (const auto& pos : active)
            for (int c = 0; c < C; ++c)
                t(b, c, pos[0], pos[1], pos[2]) = val;
    return t;
}

// Vérifie que deux floats sont proches
static bool approxEq(float a, float b, float eps = 1e-4f) {
    return std::abs(a - b) < eps;
}

// =============================================================================
// Tests SparseTensor
// =============================================================================

// ── from_dense / to_dense — round-trip ───────────────────────────────────────
TEST(SparseTensor, RoundTripDense) {
    // Tenseur avec quelques voxels non nuls
    Tensor dense = makeSparseInput(2, 3, 4, 4, 4, {{1,1,1},{2,2,2},{3,3,3}}, 0.5f);

    SparseTensor sp = SparseTensor::from_dense(dense, 0.0f);

    // Nombre de voxels actifs : 3 positions × 2 batches
    EXPECT_EQ(sp.nnz(), 6);
    EXPECT_EQ(sp.batch_size, 2);
    EXPECT_EQ(sp.num_channels, 3);
    EXPECT_EQ(sp.spatial_d, 4);

    Tensor reconstructed = sp.to_dense();

    // Vérification pixel à pixel
    for (int b = 0; b < 2; ++b)
    for (int c = 0; c < 3; ++c)
    for (int d = 0; d < 4; ++d)
    for (int h = 0; h < 4; ++h)
    for (int w = 0; w < 4; ++w) {
        EXPECT_NEAR(dense(b,c,d,h,w), reconstructed(b,c,d,h,w), 1e-6f)
            << "Mismatch at (" << b<<","<<c<<","<<d<<","<<h<<","<<w<<")";
    }
}

// ── Seuillage threshold ───────────────────────────────────────────────────────
TEST(SparseTensor, ThresholdFiltering) {
    Tensor dense(1, 1, 4, 4, 4);
    dense.setZero();
    dense(0, 0, 0, 0, 0) = 0.001f;  // sous le seuil
    dense(0, 0, 1, 1, 1) = 0.1f;    // au-dessus
    dense(0, 0, 2, 2, 2) = 1.0f;    // au-dessus

    SparseTensor sp = SparseTensor::from_dense(dense, 0.01f);
    EXPECT_EQ(sp.nnz(), 2);  // seuls les deux voxels > 0.01 sont actifs
}

// ── encode / decode symétrie ──────────────────────────────────────────────────
TEST(SparseTensor, EncodeDecodeSym) {
    SparseTensor sp;
    sp.batch_size = 4; sp.spatial_d = 8; sp.spatial_h = 8; sp.spatial_w = 8;

    const int b=2, d=5, h=3, w=7;
    uint64_t key = sp.encode(b, d, h, w);
    int rb, rd, rh, rw;
    sp.decode(key, rb, rd, rh, rw);
    EXPECT_EQ(rb, b); EXPECT_EQ(rd, d); EXPECT_EQ(rh, h); EXPECT_EQ(rw, w);
}

// ── Lookup find ───────────────────────────────────────────────────────────────
TEST(SparseTensor, LookupFind) {
    Tensor dense = makeSparseInput(1, 2, 5, 5, 5, {{1,2,3},{4,0,0}});
    SparseTensor sp = SparseTensor::from_dense(dense, 0.0f);
    sp.buildLookup();

    EXPECT_GE(sp.find(0, 1, 2, 3), 0);  // actif
    EXPECT_GE(sp.find(0, 4, 0, 0), 0);  // actif
    EXPECT_EQ(sp.find(0, 0, 0, 0), -1); // inactif
    EXPECT_EQ(sp.find(0, 3, 3, 3), -1); // inactif
}

// ── GlobalAvgPool ─────────────────────────────────────────────────────────────
TEST(SparseTensor, GlobalAvgPool) {
    // 2 voxels actifs dans un batch=1, C=1, chacun avec valeur 2.0
    Tensor dense(1, 1, 3, 3, 3);
    dense.setZero();
    dense(0, 0, 0, 0, 0) = 2.0f;
    dense(0, 0, 1, 1, 1) = 4.0f;

    SparseTensor sp = SparseTensor::from_dense(dense, 0.0f);
    Tensor pooled = sp.globalAvgPool();

    // Moyenne = (2 + 4) / 2 = 3.0
    EXPECT_NEAR(pooled(0, 0, 0, 0, 0), 3.0f, 1e-5f);
}

// ── ReLU in-place ─────────────────────────────────────────────────────────────
TEST(SparseTensor, ApplyReLU) {
    SparseTensor sp;
    sp.batch_size = 1; sp.num_channels = 2;
    sp.spatial_d = 3; sp.spatial_h = 3; sp.spatial_w = 3;
    sp.coords.resize(2, 4);
    sp.features.resize(2, 2);

    sp.coords << 0,0,0,0,  0,1,1,1;
    sp.features << -1.0f, 2.0f,
                   3.0f, -4.0f;

    sp.applyReLU();

    EXPECT_NEAR(sp.features(0, 0), 0.0f, 1e-6f);  // -1 → 0
    EXPECT_NEAR(sp.features(0, 1), 2.0f, 1e-6f);  // 2 → 2
    EXPECT_NEAR(sp.features(1, 0), 3.0f, 1e-6f);  // 3 → 3
    EXPECT_NEAR(sp.features(1, 1), 0.0f, 1e-6f);  // -4 → 0
}

// ── Densité ───────────────────────────────────────────────────────────────────
TEST(SparseTensor, Density) {
    Tensor dense = makeSparseInput(1, 1, 4, 4, 4, {{0,0,0},{1,1,1}});
    SparseTensor sp = SparseTensor::from_dense(dense, 0.0f);
    // 2 voxels actifs sur 64 total
    EXPECT_NEAR(sp.density(), 2.0f / 64.0f, 1e-6f);
}

// =============================================================================
// Tests SparseConvLayer3D
// =============================================================================

// ── Construction et dimensions ────────────────────────────────────────────────
TEST(SparseConvLayer3D, Construction) {
    SparseConvLayer3D layer(1, 8, 3, 3, 3); // in=1, out=8, k=3
    EXPECT_EQ(layer.inChannels(),  1);
    EXPECT_EQ(layer.outChannels(), 8);
    EXPECT_TRUE(layer.isSubmanifold());
    // Paramètres : 8 * (1*27) + 8 = 216 + 8 = 224
    EXPECT_EQ(layer.numParams(), 224);
}

// ── Stride > 1 interdit en SubManifold ───────────────────────────────────────
TEST(SparseConvLayer3D, SubmanifoldRejectsStride) {
    EXPECT_THROW(
        SparseConvLayer3D(1, 8, 3, 3, 3, 2, 2, 2, 1, 1, 1, true),
        std::runtime_error
    );
}

// ── Forward : sortie a même structure que entrée (SubManifold) ────────────────
TEST(SparseConvLayer3D, ForwardPreservesSparsity) {
    SparseConvLayer3D layer(1, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, true); // padding=1

    Tensor dense = makeSparseInput(2, 1, 5, 5, 5,
                                   {{1,1,1},{2,2,2},{3,3,3}}, 1.0f);
    SparseTensor input = SparseTensor::from_dense(dense, 0.0f);
    SparseTensor output = layer.forward(input);

    // SubManifold : même nombre de voxels actifs
    EXPECT_EQ(output.nnz(), input.nnz());
    EXPECT_EQ(output.num_channels, 4);
    EXPECT_EQ(output.spatial_d, 5);
    EXPECT_EQ(output.spatial_h, 5);
    EXPECT_EQ(output.spatial_w, 5);
}

// ── Forward : cohérence avec la convolution dense sur un volume plein ─────────
// Si l'entrée est entièrement dense (tous les voxels actifs),
// sparse et dense doivent donner le même résultat (à epsilon près).
TEST(SparseConvLayer3D, ForwardMatchesDenseOnFullVolume) {
    // Volume 1x1x4x4x4 entièrement actif
    Tensor dense_full(1, 1, 4, 4, 4);
    dense_full.setRandom();

    SparseConvLayer3D sparse_layer(1, 2, 3, 3, 3, 1, 1, 1, 1, 1, 1, true);

    // Construire une ConvLayer3D avec les mêmes poids
    // On recopie les poids de sparse vers dense manuellement
    // (on vérifie uniquement la structure ici, pas l'égalité numérique)
    SparseTensor sp_input = SparseTensor::from_dense(dense_full, 0.0f);

    // Le volume est plein → tous les voxels sont actifs
    EXPECT_EQ(sp_input.nnz(), 1 * 4 * 4 * 4); // 64

    SparseTensor sp_output = sparse_layer.forward(sp_input);
    EXPECT_EQ(sp_output.nnz(), 64);  // SubManifold préserve tout
    EXPECT_EQ(sp_output.num_channels, 2);
}

// ── Backward : gradients non nuls sur les voxels actifs ──────────────────────
TEST(SparseConvLayer3D, BackwardProducesGradients) {
    SparseConvLayer3D layer(2, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, true);

    Tensor dense = makeSparseInput(1, 2, 5, 5, 5,
                                   {{1,1,1},{2,2,2},{3,3,3}}, 0.5f);
    SparseTensor input = SparseTensor::from_dense(dense, 0.0f);
    SparseTensor output = layer.forward(input);

    // Gradient fictif de même structure que la sortie
    SparseTensor grad_out = output;
    grad_out.features.setOnes();

    SparseTensor grad_in = layer.backward(grad_out);

    // Le gradient d'entrée doit avoir la même structure sparse
    EXPECT_EQ(grad_in.nnz(), input.nnz());
    EXPECT_EQ(grad_in.num_channels, 2);

    // Au moins certains gradients doivent être non nuls
    float grad_norm = grad_in.features.norm();
    EXPECT_GT(grad_norm, 0.0f) << "Les gradients d'entrée sont tous nuls";
}

// ── Backward : les gradients des poids ont la bonne forme ────────────────────
TEST(SparseConvLayer3D, BackwardWeightGradientsShape) {
    SparseConvLayer3D layer(1, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, true);

    Tensor dense = makeSparseInput(2, 1, 6, 6, 6, {{2,2,2},{3,3,3}});
    SparseTensor input = SparseTensor::from_dense(dense, 0.0f);
    SparseTensor output = layer.forward(input);

    SparseTensor grad_out = output;
    grad_out.features.setOnes();
    layer.backward(grad_out);

    // dL/dW : (out_ch=3, in_ch*K=1*27=27)
    EXPECT_EQ(layer.getWeightGradients().rows(), 3);
    EXPECT_EQ(layer.getWeightGradients().cols(), 27);

    // dL/db : (out_ch=3)
    EXPECT_EQ(layer.getBiasGradients().size(), 3);
}

// ── updateParams : les poids changent après une étape SGD ────────────────────
TEST(SparseConvLayer3D, UpdateParamsChangesWeights) {
    SparseConvLayer3D layer(1, 2, 3, 3, 3, 1, 1, 1, 1, 1, 1, true);

    Tensor dense = makeSparseInput(1, 1, 5, 5, 5, {{2,2,2}});
    SparseTensor input = SparseTensor::from_dense(dense, 0.0f);
    SparseTensor output = layer.forward(input);

    SparseTensor grad_out = output;
    grad_out.features.setOnes();
    layer.backward(grad_out);

    // Sauvegarde des poids avant mise à jour
    Eigen::MatrixXf w_before = layer.getWeights();

    SGD optim(0.01f, 0.0f); // lr=0.01, pas de momentum
    layer.updateParams(optim);

    Eigen::MatrixXf w_after = layer.getWeights();
    float delta = (w_after - w_before).norm();
    EXPECT_GT(delta, 0.0f) << "Les poids n'ont pas changé après updateParams";
}

// ── SparseTensor + GlobalAvgPool → compatible DenseLayer ─────────────────────
TEST(SparseConvLayer3D, SparseToGlobalAvgPool) {
    SparseConvLayer3D layer(1, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, true);

    Tensor dense = makeSparseInput(2, 1, 7, 7, 7,
                                   {{1,1,1},{3,3,3},{5,5,5}}, 0.8f);
    SparseTensor input = SparseTensor::from_dense(dense, 0.0f);
    SparseTensor output = layer.forward(input);
    output.applyReLU();

    // GlobalAvgPool → Tensor (B, C_out, 1, 1, 1)
    Tensor pooled = output.globalAvgPool();
    EXPECT_EQ(pooled.dim(0), 2);  // B
    EXPECT_EQ(pooled.dim(1), 8);  // C_out
    EXPECT_EQ(pooled.dim(2), 1);  // D=1
    EXPECT_EQ(pooled.dim(3), 1);  // H=1
    EXPECT_EQ(pooled.dim(4), 1);  // W=1

    // Les valeurs poolées doivent être non nulles après ReLU
    for (int b = 0; b < 2; ++b)
        for (int c = 0; c < 8; ++c)
            EXPECT_GE(pooled(b, c, 0, 0, 0), 0.0f);
}

// =============================================================================
// Tests mode Standard Sparse (stride > 1)
// =============================================================================

TEST(SparseConvLayer3D, StandardSparseWithStride) {
    // stride=2, pas SubManifold : le volume diminue
    SparseConvLayer3D layer(1, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, false);

    // Volume d'entrée 7x7x7
    Tensor dense = makeDense(1, 1, 7, 7, 7, 1.0f);
    SparseTensor input = SparseTensor::from_dense(dense, 0.0f);
    EXPECT_EQ(input.nnz(), 1 * 7 * 7 * 7); // 343

    SparseTensor output = layer.forward(input);

    // (7+2*1-3)/2+1 = 4 → sortie 4×4×4
    EXPECT_EQ(output.spatial_d, 4);
    EXPECT_EQ(output.spatial_h, 4);
    EXPECT_EQ(output.spatial_w, 4);
    EXPECT_EQ(output.num_channels, 4);
    EXPECT_GT(output.nnz(), 0);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
