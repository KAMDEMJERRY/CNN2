# include "ConvLayer.hpp"
# include <cmath>
# include <random>


// ConvLayer::ConvLayer(int in_channels, int out_channels, 
//                      int kernel_size, int stride, int padding)
//     : in_channels(in_channels), out_channels(out_channels),
//       kernel_h(kernel_size), kernel_w(kernel_size),
//       stride(stride), pad_h(padding), pad_w(padding) {
    
//     // Initialiser les poids [out_channels, in_channels, kernel_h, kernel_w]
//     weights = Tensor(out_channels, in_channels, kernel_h, kernel_w);
//     bias = Eigen::VectorXf(out_channels);
    
//     initializeWeights();
// }

ConvLayer::ConvLayer(int in_channels, int out_channels,
                     int kernel_h, int kernel_w,
                     int stride_h, int stride_w,
                     int pad_h, int pad_w)
    : in_channels(in_channels), out_channels(out_channels),
      kernel_h(kernel_h), kernel_w(kernel_w),
      stride(stride_h), pad_h(pad_h), pad_w(pad_w) {

    isTrainable = true;
    
    weights = Tensor(out_channels, in_channels, kernel_h, kernel_w);
    bias = Eigen::VectorXf(out_channels);
    
    initializeWeights("he");
}

void ConvLayer::initializeWeights(const std::string& method) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    float scale;
    if (method == "he") {
        // He initialization for ReLU
        scale = std::sqrt(2.0f / (in_channels * kernel_h * kernel_w))  ;
    } else {
        // Xavier/Glorot initialization
        scale = std::sqrt(1.0f / (in_channels * kernel_h * kernel_w))  ;
    }
    
    std::normal_distribution<float> dist(0.0f, scale);
    
    // Initialiser les poids
    for (int i = 0; i < weights.size(); ++i) {
        weights[i] = dist(gen);
    }
    
    // Initialiser les biais à 0
    bias.setZero();
}


Eigen::MatrixXf ConvLayer::im2col(const Tensor& input) const {
    int batch_size = input.dim(0);
    int in_channels = input.dim(1);
    int height = input.dim(2);
    int width = input.dim(3);

    std::pair<int, int> output_dims = computeOutputDims(height, width);
    int out_height = output_dims.first;
    int out_width = output_dims.second;

    // Taille d'un patch aplatit
    int patch_size = in_channels * kernel_h * kernel_w;
    // Nombre total de patches
    int num_patch = batch_size * out_height * out_width;

    Eigen::MatrixXf col_matrix(patch_size, num_patch);

    // Remplissage de la matrice col
# pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
               
                int patch_idx = b * out_height * out_width + oh * out_width + ow;

                for (int c = 0; c < in_channels; ++c) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                         
                            int ih = oh * stride + kh - pad_h;
                            int iw = ow * stride + kw - pad_w;

                            int row_idx = c * kernel_h * kernel_w + kh * kernel_w + kw;

                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                col_matrix(row_idx, patch_idx) = input(b, c, ih, iw);
                            }
                            else {
                                col_matrix(row_idx, patch_idx) = 0.0f; // Padding zero
                            }
                        }
                    }
                }
            }
        }
    }

    return col_matrix;
    
}


Tensor ConvLayer::forward(const Tensor& input) {
    input_cache = input; // Cache pour le backward pass

    int batch_size = input.dim(0);
    int height = input.dim(2);
    int width = input.dim(3);

    std::pair<int, int> output_dims = computeOutputDims(height, width);
    int out_height = output_dims.first;
    int out_width = output_dims.second;

    // Convertir les poids en matrice
    // weights_matrix: [out_channels] x [in_channels * kernel_h * kernel_w]
    Eigen::MatrixXf weights_matrix(out_channels, in_channels * kernel_h * kernel_w);
    convertWeightsToMatrix(weights_matrix);
    

    // Appliquer im2col
    col_cache = im2col(input);

    // Multiplication matricielle avec Eigen  (optimisee)
    Eigen::MatrixXf output_matrix = weights_matrix * col_cache;

    // Ajouter le biais
    output_matrix.colwise() += bias;

    // Reorganiser en tensor 4D
    Tensor output(batch_size, out_channels, out_height, out_width);

# pragma omp parallel for collapse(4)
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    int col_idx = b * out_height * out_width + oh * out_width + ow;
                    output(b, oc, oh, ow) = output_matrix(oc, col_idx);
                }
            }
        }
    }

    return output;
}


Tensor ConvLayer::backward(const Tensor& gradOutput) {
    int batch_size = input_cache.dim(0);
    int height = input_cache.dim(2);
    int width = input_cache.dim(3);

    std::pair<int, int> output_dims = computeOutputDims(height, width);
    int out_height = output_dims.first;
    int out_width = output_dims.second;

    // Convertir grad_output en matrice
    Eigen::MatrixXf grad_output_matrix = gradOutputToMatrix(gradOutput, out_height, out_width);

    // Calculer le gradient des poids
    // dL/dW = dL/dY * im2col(X)^T
    Eigen::MatrixXf grad_weights_matrix = grad_output_matrix * col_cache.transpose();   
    grad_weights_matrix /= static_cast<float>(batch_size);
    convertMatrixToWeights(grad_weights_matrix);

    // Gradient par rapport aux biais
    grad_bias = grad_output_matrix.rowwise().sum();
    grad_bias /= static_cast<float>(batch_size* out_height * out_width);

    // Gradient par rapport à l'entrée
    Eigen::MatrixXf weights_matrix(out_channels, in_channels * kernel_h * kernel_w);
    convertWeightsToMatrix(weights_matrix);

    Eigen::MatrixXf grad_input_matrix = weights_matrix.transpose() * grad_output_matrix;
    Tensor grad_input = col2im(grad_input_matrix, height, width);

    col_cache = Eigen::MatrixXf(); // Clear cache to save memory

    return grad_input;  
}

// Convertir col2im (l'inverse d'im2col) - C'est la partie la plus complexe
Tensor ConvLayer::col2im(const Eigen::MatrixXf& col_matrix, int height, int width) const {
    int batch_size = input_cache.dim(0);
    auto [out_height, out_width] = computeOutputDims(height, width);
    
    Tensor grad_input(batch_size, in_channels, height, width);
    grad_input.setZero();
    
    // Pour chaque élément dans la matrice col, accumuler dans le bon pixel
    for (int b = 0; b < batch_size; ++b) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                int col_idx = b * out_height * out_width + oh * out_width + ow;
                
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int ih = oh * stride + kh - pad_h;
                            int iw = ow * stride + kw - pad_w;
                            
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                int row_idx = ic * kernel_h * kernel_w + kh * kernel_w + kw;
                                // Accumuler les gradients (somme sur toutes les positions)
                                grad_input(b, ic, ih, iw) += col_matrix(row_idx, col_idx);
                            }
                        }
                    }
                }
            }
        }
    }
    
    return grad_input;
}


// Convertir grad_output en matrice
Eigen::MatrixXf ConvLayer::gradOutputToMatrix(const Tensor& grad_output, int out_height, int out_width) const {
    int batch_size = grad_output.dim(0);
    Eigen::MatrixXf grad_output_matrix(out_channels, batch_size * out_height * out_width);
    
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; ++b) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                int col_idx = b * out_height * out_width + oh * out_width + ow;
                for (int oc = 0; oc < out_channels; ++oc) {
                    grad_output_matrix(oc, col_idx) = grad_output(b, oc, oh, ow);
                }
            }
        }
    }
    
    return grad_output_matrix;
}