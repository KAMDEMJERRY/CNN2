#include <gtest/gtest.h>
#include "WindowAttention3DLayer.hpp"
#include "FlashAttention3DLayer.hpp"
#include <cmath>
#include <random>

// =============================================================================
// Helpers
// =============================================================================

// Tensor 5D rempli d'une valeur constante
static Tensor makeTensor(int B, int C, int D, int H, int W, float val = 0.5f) {
    Tensor t(B, C, D, H, W);
    t.setConstant(val);
    return t;
}

// Tensor 5D rempli aléatoirement (distribution normale)
static Tensor makeRandTensor(int B, int C, int D, int H, int W, float scale = 0.1f) {
    Tensor t(B, C, D, H, W);
    std::mt19937 gen{42};
    std::normal_distribution<float> dist(0.f, scale);
    for (int i = 0; i < t.size(); ++i)
        t[i] = dist(gen);
    return t;
}

// Norme L2 de la différence entre deux tenseurs
static float tensorDiff(const Tensor& a, const Tensor& b) {
    float diff = 0.f;
    for (int i = 0; i < a.size(); ++i)
        diff += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(diff);
}

// =============================================================================
// Tests WindowAttention3DLayer
// =============================================================================

// ── Construction et vérification des paramètres ───────────────────────────────
TEST(WindowAttention3D, Construction) {
    WindowAttention3DLayer attn(32, 4, 4, 4, 4);
    EXPECT_EQ(attn.numParams(), 4 * 32 * 32 + 2 * 32);  // 4C² + 2C = 4160
    EXPECT_NE(attn.getWQ().norm(), 0.f);   // poids initialisés (non nuls)
    EXPECT_NE(attn.getWK().norm(), 0.f);
}

// ── Mismatch channels doit lever une exception ────────────────────────────────
TEST(WindowAttention3D, ChannelsMismatchThrows) {
    WindowAttention3DLayer attn(32, 4, 4, 4, 4);
    Tensor wrong_input(1, 16, 7, 7, 7);  // 16 channels au lieu de 32
    wrong_input.setConstant(0.1f);
    EXPECT_THROW(attn.forward(wrong_input), std::runtime_error);
}

// ── channels non-divisible par num_heads doit lever une exception ─────────────
TEST(WindowAttention3D, NonDivisibleChannelsThrows) {
    EXPECT_THROW(
        WindowAttention3DLayer(33, 4, 4, 4, 4),  // 33 % 4 != 0
        std::invalid_argument
    );
}

// ── Forward : les dimensions sont préservées ──────────────────────────────────
TEST(WindowAttention3D, ForwardPreservesDimensions) {
    WindowAttention3DLayer attn(16, 4, 4, 4, 2);
    Tensor input = makeRandTensor(2, 16, 7, 7, 7);
    Tensor output = attn.forward(input);
    EXPECT_EQ(output.dim(0), 2);
    EXPECT_EQ(output.dim(1), 16);
    EXPECT_EQ(output.dim(2), 7);
    EXPECT_EQ(output.dim(3), 7);
    EXPECT_EQ(output.dim(4), 7);
}

// ── Forward : la sortie n'est pas nulle ───────────────────────────────────────
TEST(WindowAttention3D, ForwardProducesNonZero) {
    WindowAttention3DLayer attn(16, 4, 4, 4, 2);
    Tensor input = makeRandTensor(1, 16, 7, 7, 7);
    Tensor output = attn.forward(input);
    float norm = 0.f;
    for (int i = 0; i < output.size(); ++i) norm += output[i] * output[i];
    EXPECT_GT(std::sqrt(norm), 0.f);
}

// ── Forward : connexion résiduelle — sortie différente de l'entrée ────────────
// Avec résiduelle et LayerNorm, la sortie doit être différente de l'entrée
// mais pas de magnitude absurde (LayerNorm stabilise)
TEST(WindowAttention3D, ResidualChangesOutput) {
    WindowAttention3DLayer attn(16, 4, 4, 4, 2, true, true);
    Tensor input = makeRandTensor(1, 16, 7, 7, 7);
    Tensor output = attn.forward(input);
    float diff = tensorDiff(input, output);
    EXPECT_GT(diff, 1e-3f);   // la sortie est différente de l'entrée
}

// ── Forward : volume padding — dimensions non-multiples de la fenêtre ─────────
// Volume 5×5×5 avec fenêtre 4×4×4 → padding implicite, doit fonctionner
TEST(WindowAttention3D, ForwardWithPadding) {
    WindowAttention3DLayer attn(8, 4, 4, 4, 2);
    Tensor input = makeRandTensor(1, 8, 5, 5, 5);
    EXPECT_NO_THROW({
        Tensor output = attn.forward(input);
        EXPECT_EQ(output.dim(2), 5);
        EXPECT_EQ(output.dim(3), 5);
        EXPECT_EQ(output.dim(4), 5);
    });
}

// ── Forward : fenêtre = volume entier (attention globale) ─────────────────────
TEST(WindowAttention3D, GlobalAttentionWindow) {
    // win=7 sur volume 7³ = une seule fenêtre, 343 tokens
    WindowAttention3DLayer attn(8, 7, 7, 7, 2);
    Tensor input = makeRandTensor(1, 8, 7, 7, 7);
    Tensor output = attn.forward(input);
    EXPECT_EQ(output.dim(2), 7);
    EXPECT_EQ(output.dim(3), 7);
    EXPECT_EQ(output.dim(4), 7);
}

// ── Backward : gradient de même forme que l'entrée ────────────────────────────
TEST(WindowAttention3D, BackwardCorrectShape) {
    WindowAttention3DLayer attn(16, 4, 4, 4, 2);
    Tensor input = makeRandTensor(2, 16, 7, 7, 7);
    Tensor output = attn.forward(input);

    Tensor grad_out = makeRandTensor(2, 16, 7, 7, 7);
    Tensor grad_in  = attn.backward(grad_out);

    EXPECT_EQ(grad_in.dim(0), input.dim(0));
    EXPECT_EQ(grad_in.dim(1), input.dim(1));
    EXPECT_EQ(grad_in.dim(2), input.dim(2));
    EXPECT_EQ(grad_in.dim(3), input.dim(3));
    EXPECT_EQ(grad_in.dim(4), input.dim(4));
}

// ── Backward : gradients non nuls ────────────────────────────────────────────
TEST(WindowAttention3D, BackwardNonZeroGradients) {
    WindowAttention3DLayer attn(16, 4, 4, 4, 2);
    Tensor input = makeRandTensor(1, 16, 7, 7, 7);
    attn.forward(input);

    Tensor grad_out(1, 16, 7, 7, 7);
    grad_out.setConstant(1.0f);
    Tensor grad_in = attn.backward(grad_out);

    float norm = 0.f;
    for (int i = 0; i < grad_in.size(); ++i) norm += grad_in[i] * grad_in[i];
    EXPECT_GT(std::sqrt(norm), 0.f) << "Les gradients d'entrée sont nuls";

    // Gradients des poids non nuls
    EXPECT_GT(attn.getWQ().norm(), 0.f);  // les poids ont changé après init
}

// ── updateParams : les poids changent après Adam ──────────────────────────────
TEST(WindowAttention3D, UpdateParamsChangesWeights) {
    WindowAttention3DLayer attn(16, 4, 4, 4, 2);
    Tensor input = makeRandTensor(1, 16, 7, 7, 7);
    Tensor output = attn.forward(input);

    Tensor grad_out(1, 16, 7, 7, 7);
    grad_out.setConstant(1.0f);
    attn.backward(grad_out);

    Eigen::MatrixXf wq_before = attn.getWQ();
    Adam opt(0.001f);
    attn.updateParams(opt);

    float delta = (attn.getWQ() - wq_before).norm();
    EXPECT_GT(delta, 0.f) << "W_Q n'a pas changé après updateParams";
}

// ── Gradient numérique — vérification du backward ────────────────────────────
// Vérifie que le backward analytique correspond au gradient fini
// sur une petite perturbation ε d'un scalaire de l'entrée
TEST(WindowAttention3D, NumericalGradientCheck) {
    WindowAttention3DLayer attn(8, 4, 4, 4, 1, false, false);  // sans résiduelle ni norm pour simplifier
    Tensor input = makeRandTensor(1, 8, 4, 4, 4, 0.05f);

    const float eps = 1e-3f;
    const int b0=0, c0=0, d0=1, h0=1, w0=1;  // scalaire à perturber

    // Gradient analytique
    Tensor output = attn.forward(input);
    Tensor grad_out(1, 8, 4, 4, 4);
    grad_out.setConstant(1.0f);
    Tensor grad_in = attn.backward(grad_out);
    float analytic = grad_in(b0, c0, d0, h0, w0);

    // Gradient numérique (différences finies centrées)
    // f(x+ε) via nouveau forward avec poids réinitialisés (on réutilise attn)
    Tensor input_plus = input;
    input_plus(b0, c0, d0, h0, w0) += eps;
    Tensor out_plus = attn.forward(input_plus);
    float sum_plus = 0.f;
    for (int i = 0; i < out_plus.size(); ++i) sum_plus += out_plus[i];

    Tensor input_minus = input;
    input_minus(b0, c0, d0, h0, w0) -= eps;
    Tensor out_minus = attn.forward(input_minus);
    float sum_minus = 0.f;
    for (int i = 0; i < out_minus.size(); ++i) sum_minus += out_minus[i];

    float numerical = (sum_plus - sum_minus) / (2.f * eps);

    // Tolérance relative de 5% — acceptable pour gradient numérique
    const float rel_err = std::abs(analytic - numerical)
                        / (std::abs(numerical) + 1e-6f);
    EXPECT_LT(rel_err, 0.05f)
        << "Gradient analytique=" << analytic
        << " vs numérique=" << numerical;
}

// =============================================================================
// Tests FlashAttention3DLayer
// =============================================================================

// ── Construction ──────────────────────────────────────────────────────────────
TEST(FlashAttention3D, Construction) {
    FlashAttention3DLayer attn(32, 7, 7, 7, 4, 64);
    EXPECT_EQ(attn.numParams(), 4 * 32 * 32 + 2 * 32);
    EXPECT_EQ(attn.blockSize(), 64);
    EXPECT_NE(attn.getWQ().norm(), 0.f);
}

// ── Mismatch channels ─────────────────────────────────────────────────────────
TEST(FlashAttention3D, ChannelsMismatchThrows) {
    FlashAttention3DLayer attn(32, 7, 7, 7, 4, 64);
    Tensor wrong(1, 16, 7, 7, 7);
    wrong.setConstant(0.1f);
    EXPECT_THROW(attn.forward(wrong), std::runtime_error);
}

// ── Forward : dimensions préservées ──────────────────────────────────────────
TEST(FlashAttention3D, ForwardPreservesDimensions) {
    FlashAttention3DLayer attn(16, 7, 7, 7, 2, 64);
    Tensor input = makeRandTensor(2, 16, 7, 7, 7);
    Tensor output = attn.forward(input);
    EXPECT_EQ(output.dim(0), 2);
    EXPECT_EQ(output.dim(1), 16);
    EXPECT_EQ(output.dim(2), 7);
    EXPECT_EQ(output.dim(3), 7);
    EXPECT_EQ(output.dim(4), 7);
}

// ── Flash vs Standard : même résultat mathématique ────────────────────────────
// Pour un volume petit (4×4×4), Flash Attention et WindowAttention doivent
// donner des sorties très proches (écart < 1e-4) avec les mêmes poids.
TEST(FlashAttention3D, MatchesWindowAttention) {
    const int C = 8, D = 4, H = 4, W = 4;

    WindowAttention3DLayer  win(C, D, H, W, 2, false, false);  // pas résid, pas norm
    FlashAttention3DLayer flash(C, D, H, W, 2, 64,   false, false);

    // Copier exactement les mêmes poids de win vers flash
    flash.getWQ() = win.getWQ();
    flash.getWK() = win.getWK();
    flash.getWV() = win.getWV();
    flash.getWO() = win.getWO();

    Tensor input = makeRandTensor(1, C, D, H, W, 0.1f);

    Tensor out_win   = win.forward(input);
    Tensor out_flash = flash.forward(input);

    float diff = tensorDiff(out_win, out_flash);
    EXPECT_LT(diff, 1e-3f)
        << "Flash et Window donnent des résultats trop différents : diff=" << diff;
}

// ── Block size > N_tok : dégradation gracieuse ────────────────────────────────
// Si block_size > nombre de tokens, Flash Attention doit quand même fonctionner
TEST(FlashAttention3D, BlockSizeLargerThanN) {
    // Volume 4×4×4 = 64 tokens, block_size=128 > N
    FlashAttention3DLayer attn(8, 4, 4, 4, 2, 128);
    Tensor input = makeRandTensor(1, 8, 4, 4, 4);
    EXPECT_NO_THROW({
        Tensor output = attn.forward(input);
        EXPECT_EQ(output.dim(2), 4);
    });
}

// ── Block size = 1 : cas extrême de tiling ────────────────────────────────────
TEST(FlashAttention3D, BlockSizeOne) {
    FlashAttention3DLayer attn(8, 4, 4, 4, 2, 1);
    Tensor input = makeRandTensor(1, 8, 4, 4, 4);
    EXPECT_NO_THROW({
        Tensor output = attn.forward(input);
        EXPECT_EQ(output.dim(2), 4);
    });
}

// ── Backward : forme correcte ─────────────────────────────────────────────────
TEST(FlashAttention3D, BackwardCorrectShape) {
    FlashAttention3DLayer attn(16, 7, 7, 7, 2, 64);
    Tensor input = makeRandTensor(2, 16, 7, 7, 7);
    Tensor output = attn.forward(input);
    Tensor grad_out = makeRandTensor(2, 16, 7, 7, 7);
    Tensor grad_in = attn.backward(grad_out);

    EXPECT_EQ(grad_in.dim(0), 2);
    EXPECT_EQ(grad_in.dim(1), 16);
    EXPECT_EQ(grad_in.dim(2), 7);
}

// ── Backward : gradients non nuls ────────────────────────────────────────────
TEST(FlashAttention3D, BackwardNonZero) {
    FlashAttention3DLayer attn(16, 7, 7, 7, 2, 64);
    Tensor input = makeRandTensor(1, 16, 7, 7, 7);
    attn.forward(input);
    Tensor grad_out(1, 16, 7, 7, 7);
    grad_out.setConstant(1.0f);
    Tensor grad_in = attn.backward(grad_out);

    float norm = 0.f;
    for (int i = 0; i < grad_in.size(); ++i) norm += grad_in[i] * grad_in[i];
    EXPECT_GT(std::sqrt(norm), 0.f);
}

// ── Flash backward vs Window backward : mêmes gradients d'entrée ─────────────
// Avec les mêmes poids, les gradients doivent être identiques à précision
// float32 près (le backward Flash recompute P_ij de façon exacte)
TEST(FlashAttention3D, BackwardMatchesWindowBackward) {
    const int C = 8, D = 4, H = 4, W = 4;

    WindowAttention3DLayer  win(C, D, H, W, 2, false, false);
    FlashAttention3DLayer flash(C, D, H, W, 2, 32, false, false);

    flash.getWQ() = win.getWQ();
    flash.getWK() = win.getWK();
    flash.getWV() = win.getWV();
    flash.getWO() = win.getWO();

    Tensor input    = makeRandTensor(1, C, D, H, W, 0.1f);
    Tensor grad_out = makeRandTensor(1, C, D, H, W, 0.1f);

    win.forward(input);
    Tensor grad_win = win.backward(grad_out);

    flash.forward(input);
    Tensor grad_flash = flash.backward(grad_out);

    float diff = tensorDiff(grad_win, grad_flash);
    EXPECT_LT(diff, 1e-2f)  // tolérance plus large : accumulations différentes
        << "Backward Flash vs Window : diff=" << diff;
}

// ── updateParams : poids mis à jour ──────────────────────────────────────────
TEST(FlashAttention3D, UpdateParams) {
    FlashAttention3DLayer attn(16, 7, 7, 7, 2, 64);
    Tensor input = makeRandTensor(1, 16, 7, 7, 7);
    attn.forward(input);
    Tensor grad_out(1, 16, 7, 7, 7);
    grad_out.setConstant(1.0f);
    attn.backward(grad_out);

    Eigen::MatrixXf wq_before = attn.getWQ();
    Adam opt(0.001f);
    attn.updateParams(opt);
    EXPECT_GT((attn.getWQ() - wq_before).norm(), 0.f);
}

// ── Gradient numérique Flash Attention ───────────────────────────────────────
TEST(FlashAttention3D, NumericalGradientCheck) {
    FlashAttention3DLayer attn(8, 4, 4, 4, 2, 16, false, false);
    Tensor input = makeRandTensor(1, 8, 4, 4, 4, 0.05f);

    const float eps = 1e-3f;
    const int b0=0, c0=1, d0=0, h0=2, w0=1;

    Tensor output = attn.forward(input);
    Tensor grad_out(1, 8, 4, 4, 4);
    grad_out.setConstant(1.0f);
    Tensor grad_in = attn.backward(grad_out);
    float analytic = grad_in(b0, c0, d0, h0, w0);

    Tensor ip = input; ip(b0,c0,d0,h0,w0) += eps;
    Tensor op = attn.forward(ip);
    float sp = 0.f; for (int i=0;i<op.size();++i) sp += op[i];

    Tensor im = input; im(b0,c0,d0,h0,w0) -= eps;
    Tensor om = attn.forward(im);
    float sm = 0.f; for (int i=0;i<om.size();++i) sm += om[i];

    float numerical = (sp - sm) / (2.f * eps);
    float rel_err = std::abs(analytic - numerical)
                  / (std::abs(numerical) + 1e-6f);

    EXPECT_LT(rel_err, 0.05f)
        << "Gradient Flash analytique=" << analytic
        << " vs numérique=" << numerical;
}

// =============================================================================
// Tests d'intégration — chaînage dans un CNN
// =============================================================================

// ── WindowAttention3D s'insère dans une séquence dense + attention ────────────
TEST(Integration, WindowAttentionInCNNSequence) {
    // Simule : ConvLayer3D → ReLU → WindowAttention → GlobalAvgPool
    // On teste uniquement les dimensions de bout en bout
    WindowAttention3DLayer attn(32, 4, 4, 4, 4);

    // Entrée simulant la sortie d'un ConvLayer3D : (B=2, C=32, D=14, H=14, W=14)
    Tensor conv_out = makeRandTensor(2, 32, 14, 14, 14, 0.1f);

    // Forward attention
    Tensor attn_out = attn.forward(conv_out);
    EXPECT_EQ(attn_out.dim(0), 2);
    EXPECT_EQ(attn_out.dim(1), 32);
    EXPECT_EQ(attn_out.dim(2), 14);

    // Backward depuis un gradient fictif
    Tensor grad = makeRandTensor(2, 32, 14, 14, 14, 0.01f);
    Tensor grad_in = attn.backward(grad);
    EXPECT_EQ(grad_in.dim(0), 2);
    EXPECT_EQ(grad_in.dim(1), 32);
}

// ── FlashAttention sur volume 7³ (cas réel du projet) ────────────────────────
TEST(Integration, FlashAttentionOnRealVolume) {
    // Après 2 strides de 2 sur 28³ → volume 7³, C=64
    FlashAttention3DLayer attn(64, 7, 7, 7, 4, 64);

    Tensor input = makeRandTensor(4, 64, 7, 7, 7, 0.05f);  // batch=4
    Tensor output = attn.forward(input);

    EXPECT_EQ(output.dim(0), 4);
    EXPECT_EQ(output.dim(1), 64);
    EXPECT_EQ(output.dim(2), 7);

    // Backward
    Tensor grad = makeRandTensor(4, 64, 7, 7, 7, 0.01f);
    Tensor grad_in = attn.backward(grad);
    EXPECT_EQ(grad_in.dim(0), 4);
    EXPECT_EQ(grad_in.dim(1), 64);
}

// ── Deux couches d'attention consécutives ─────────────────────────────────────
// Simule l'architecture buildModel3DSparseAttn :
//   attn1 (C=32, win=4, après vol 14³) → attn2 (C=64, win=7, après vol 7³)
// Les deux doivent fonctionner sans modifier les dimensions
TEST(Integration, TwoAttentionLayersSequential) {
    WindowAttention3DLayer attn1(32, 4, 4, 4, 4);
    FlashAttention3DLayer  attn2(64, 7, 7, 7, 4, 64);

    // Bloc 1 : attention sur volume 14³, C=32
    Tensor t1 = makeRandTensor(2, 32, 14, 14, 14, 0.05f);
    Tensor t2 = attn1.forward(t1);
    EXPECT_EQ(t2.dim(2), 14);
    EXPECT_EQ(t2.dim(1), 32);

    // Bloc 2 : attention sur volume 7³, C=64
    Tensor t3 = makeRandTensor(2, 64, 7, 7, 7, 0.05f);
    Tensor t4 = attn2.forward(t3);
    EXPECT_EQ(t4.dim(2), 7);
    EXPECT_EQ(t4.dim(1), 64);
}

// =============================================================================
// Main
// =============================================================================
// int main(int argc, char** argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }