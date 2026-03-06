#include "DenseLayer.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// Constructeur
// ─────────────────────────────────────────────────────────────────────────────
DenseLayer::DenseLayer(int input_size, int output_size)
    : input_size(input_size), output_size(output_size)
    , weights    (Eigen::MatrixXf(output_size, input_size))
    , bias       (Eigen::VectorXf::Zero(output_size))
    , grad_weights(Eigen::MatrixXf::Zero(output_size, input_size))
    , grad_bias  (Eigen::VectorXf::Zero(output_size))
{
    isTrainable = true;
    initializeWeights("xavier");
}

// ─────────────────────────────────────────────────────────────────────────────
// initializeWeights — identique à votre version, random_device comme ConvLayer
// ─────────────────────────────────────────────────────────────────────────────
void DenseLayer::initializeWeights(const std::string& method) {
    std::random_device rd;
    std::mt19937 gen(rd());

    float scale;
    if (method == "xavier") {
        scale = std::sqrt(2.0f / (input_size + output_size));
    } else if (method == "he") {
        scale = std::sqrt(2.0f / input_size);
    } else {
        scale = 0.1f;
    }

    std::normal_distribution<float> dist(0.0f, scale);

    for (int i = 0; i < weights.rows(); ++i)
        for (int j = 0; j < weights.cols(); ++j)
            weights(i, j) = dist(gen);

    bias.setZero();
    grad_weights.setZero();
    grad_bias.setZero();
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward
//
// 1. toMatrix() : aplatit (B, C, [D,] H, W) → (B, input_size)
//                 fonctionne en 4D et 5D grâce au Tensor unifié
// 2. GEMM      : (B, input_size) × (input_size, output_size) → (B, output_size)
// 3. Bias      : rowwise()
// 4. Sortie    : (B, output_size, 1, 1) ou (B, output_size, 1, 1, 1)
//                selon le rang logique de l'entrée
// ─────────────────────────────────────────────────────────────────────────────
Tensor DenseLayer::forward(const Tensor& input) {
    // Conserver le rang logique pour reconstruire la sortie et le grad_input
    cached_rank = input.ndim();
    input_cache = input;

    // Flatten → (B, input_size)
    // toMatrix() du Tensor unifié gère 4D et 5D
    Eigen::MatrixXf input_matrix = input.toMatrix();

    if (input_matrix.cols() != input_size)
        throw std::runtime_error(
            "[DenseLayer] forward: input_size mismatch — "
            "attendu " + std::to_string(input_size) +
            ", reçu "  + std::to_string(input_matrix.cols()) +
            ". Vérifiez GlobalAvgPool ou Flatten avant DenseLayer.");

    // GEMM : (B, input_size) × (input_size, output_size)
    Eigen::MatrixXf output_matrix = input_matrix * weights.transpose();

    // Biais
    output_matrix.rowwise() += bias.transpose();

    // Reconstruire dans le bon rang
    return buildOutput(output_matrix, static_cast<int>(input_matrix.rows()));
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward
//
// grad_output : (B, output_size, 1, 1) ou (B, output_size, 1, 1, 1)
//
// 1. Flatten grad_output    → (B, output_size)
// 2. grad_weights = (input^T × grad_output)^T / B   → (output_size, input_size)
// 3. grad_bias    = sum(grad_output, axis=0) / B     → (output_size)
// 4. grad_input   = grad_output × weights            → (B, input_size)
//                   reshapé dans la shape originale de l'entrée
// ─────────────────────────────────────────────────────────────────────────────
Tensor DenseLayer::backward(const Tensor& grad_output) {
    Eigen::MatrixXf dO = grad_output.toMatrix();   // (B, output_size)
    Eigen::MatrixXf X  = input_cache.toMatrix();   // (B, input_size)

    int B = static_cast<int>(dO.rows());

    // 1. Gradient des poids — même normalisation que ConvLayer : / batch_size
    grad_weights = (X.transpose() * dO).transpose();
    grad_weights /= static_cast<float>(B);

    // 2. Gradient des biais
    grad_bias = dO.colwise().sum().transpose();
    grad_bias /= static_cast<float>(B);

    // 3. Gradient vers l'entrée : (B, output_size) × (output_size, input_size)
    Eigen::MatrixXf dX = dO * weights;   // (B, input_size)

    // 4. Reconstruire dans la shape et le rang de l'entrée originale
    return buildGradInput(dX);
}

// ─────────────────────────────────────────────────────────────────────────────
// Setters
// ─────────────────────────────────────────────────────────────────────────────
void DenseLayer::setWeights(const Eigen::MatrixXf& new_weights) {
    if (new_weights.rows() != output_size || new_weights.cols() != input_size)
        throw std::runtime_error(
            "[DenseLayer] setWeights: dimensions incorrectes — "
            "attendu (" + std::to_string(output_size) + ", " +
            std::to_string(input_size) + ")");
    weights = new_weights;
}

void DenseLayer::setBias(const Eigen::VectorXf& new_bias) {
    if (new_bias.size() != output_size)
        throw std::runtime_error(
            "[DenseLayer] setBias: taille incorrecte — "
            "attendu " + std::to_string(output_size));
    bias = new_bias;
}