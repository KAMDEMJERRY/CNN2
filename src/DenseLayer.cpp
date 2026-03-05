// DenseLayer.cpp
# include "DenseLayer.hpp"
# include <cmath>
# include <random>
# include <stdexcept>
# include <iostream>

// Constructeur
DenseLayer::DenseLayer(int input_size, int output_size)
    : input_size(input_size), output_size(output_size) {

    isTrainable = true;
    // Initialiser les matrices
    weights = Eigen::MatrixXf(output_size, input_size);
    bias = Eigen::VectorXf(output_size);

    initializeWeights("he");
}

// Initialisation des poids
void DenseLayer::initializeWeights(const std::string& method) {
    std::random_device rd;
    std::mt19937 gen(rd());

    float scale;
    if (method == "xavier") {
        // Xavier/Glorot initialization
        scale = std::sqrt(2.0f / (input_size + output_size)) ;
    }
    else if (method == "he") {
        // He initialization (bon pour ReLU)
        scale = std::sqrt(2.0f / input_size) ;
    }
    else {
        // Uniform initialization
        scale = 0.1f;
    }

    std::normal_distribution<float> dist(0.0f, scale);

    // Initialiser les poids
    for (int i = 0; i < weights.rows(); ++i) {
        for (int j = 0; j < weights.cols(); ++j) {
            weights(i, j) = dist(gen);
        }
    }

    // Initialiser les biais à 0
    bias.setZero();
}

// Convertir Tensor en Eigen Matrix
Eigen::MatrixXf DenseLayer::tensorToMatrix(const Tensor& tensor) const {
    int batch_size = tensor.dim(0);

    // Flatten le tensor en une matrice [batch_size x (total_elements)]
    int total_elements = 1;
    for (int i = 1; i < 4; ++i) {  // Ignorer la dimension batch
        if (tensor.dim(i) > 0) {
            total_elements *= tensor.dim(i);
        }
    }

    // Vérifier que le flatten donne la bonne taille
    // if (total_elements != input_size) {
    //     throw std::runtime_error("Taille d'entrée incompatible avec input_size de DenseLayer");
    // }

    // Créer une vue sur les données
    Eigen::Map<const Eigen::MatrixXf> tensor_map(
        tensor.getData(),
        total_elements,
        batch_size
    );

    return tensor_map.transpose();  // [batch_size x input_size]
}

// Convertir Eigen Matrix en Tensor
Tensor DenseLayer::matrixToTensor(const Eigen::MatrixXf& matrix, bool is_output_size) const {
    int batch_size = matrix.rows();
    // Créer un tensor avec la shape [batch_size, size, 1, 1]
    // Conversion en tenseur de taille outpout(forward)
    // ou en tenseur de taille input(backward)
    int size = is_output_size ? output_size : input_size;
    Tensor tensor(batch_size, size, 1, 1);

    // Copier les données
    Eigen::Map<Eigen::MatrixXf> tensor_map(
        tensor.getData(),
        size,
        batch_size
    );

    tensor_map = matrix.transpose();

    return tensor;
}

// Forward pass
Tensor DenseLayer::forward(const Tensor& input) {
    // Convertir l'entrée en matrice
    Eigen::MatrixXf input_matrix = input.toMatrix();

    // Verifier les dimensions
    if (input_matrix.cols() != input_size) {
        throw std::runtime_error("Input size mismatch");
    }

    // Sauvegarder pour la backpropagation
    input_cache = input;

    // Calcul: output = input * weights^T + bias
    Eigen::MatrixXf output_matrix = input_matrix * weights.transpose();

    // Ajouter le biais à chaque ligne
    output_matrix.rowwise() += bias.transpose();

    // Convertir en tensor de sortie
    return Tensor::fromMatrix(output_matrix, input.dim(0), output_size, 1, 1);
}

// Backward pass
// Backward pass CORRIGÉE
Tensor DenseLayer::backward(const Tensor& grad_output) {
    // Convertir en matrices
    Eigen::MatrixXf grad_output_matrix = grad_output.toMatrix();
    Eigen::MatrixXf input_matrix = input_cache.toMatrix();

    int batch_size = grad_output_matrix.rows();

    // ✅ 1. Gradient des poids - CORRIGÉ
    grad_weights = (input_matrix.transpose() * grad_output_matrix).transpose();
    grad_weights /= static_cast<float>(batch_size);

    // ✅ 2. Gradient des biais
    grad_bias = grad_output_matrix.colwise().sum();
    grad_bias /= static_cast<float>(batch_size);

    // ✅ 3. Gradient de l'entrée
    Eigen::MatrixXf grad_input_matrix = grad_output_matrix * weights;

    // ✅ 4. DEBUG - Vérifier les dimensions
    // std::cout << "DenseLayer backward - weights grad shape: "
    //     << grad_weights.rows() << "x" << grad_weights.cols()
    //     << " (should be " << output_size << "x" << input_size << ")" << std::endl;

    return Tensor::fromMatrix(grad_input_matrix, input_cache.shape());
}

// Setters
void DenseLayer::setWeights(const Eigen::MatrixXf& new_weights) {
    if (new_weights.rows() == output_size && new_weights.cols() == input_size) {
        weights = new_weights;
    }
    else {
        throw std::runtime_error("Dimensions des poids incorrectes");
    }
}

void DenseLayer::setBias(const Eigen::VectorXf& new_bias) {
    if (new_bias.size() == output_size) {
        bias = new_bias;
    }
    else {
        throw std::runtime_error("Dimension du biais incorrecte");
    }
}