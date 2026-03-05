# include "MaxPoolLayer.hpp"


Tensor MaxPoolLayer::forward(const Tensor& input) {
    input_cache = input; // Sauvegarde pour le backward
    
    int batch_size = input.dim(0);
    int channels = input.dim(1);
    int height = input.dim(2);
    int width = input.dim(3);
    
    auto [out_height, out_width] = computeOutputDims(height, width);
    
    // Créer le tensor de sortie
    Tensor output(batch_size, channels, out_height, out_width);
    
    // Initialiser le cache des indices
    max_indices.clear();
    max_indices.resize(batch_size * channels * out_height * out_width);
    
    // #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    // Coordonnées de début dans l'input
                    int start_h = oh * stride;
                    int start_w = ow * stride;
                    
                    // Trouver le maximum dans la fenêtre de pooling
                    float max_val = std::numeric_limits<float>::lowest();
                    int max_h = start_h;
                    int max_w = start_w;
                    
                    for (int ph = 0; ph < pool_size; ++ph) {
                        for (int pw = 0; pw < pool_size; ++pw) {
                            int ih = start_h + ph;
                            int iw = start_w + pw;
                            
                            if (ih < height && iw < width) {
                                float val = input(b, c, ih, iw);
                                if (val > max_val) {
                                    max_val = val;
                                    max_h = ih;
                                    max_w = iw;
                                }
                            }
                        }
                    }
                    
                    // Stocker la valeur maximale
                    output(b, c, oh, ow) = max_val;
                    
                    // Stocker l'indice pour le backward
                    int idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                    max_indices[idx] = {max_h, max_w};
                }
            }
        }
    }
    // printf("flatten outpu size %d ", channels * out_height * out_width);
    return output;
}

Tensor MaxPoolLayer::backward(const Tensor& grad_output) {
    int batch_size = input_cache.dim(0);
    int channels = input_cache.dim(1);
    int height = input_cache.dim(2);
    int width = input_cache.dim(3);
    
    auto [out_height, out_width] = computeOutputDims(height, width);
    
    // Créer le gradient d'entrée
    Tensor grad_input(batch_size, channels, height, width);
    grad_input.setZero(); // Initialiser à zéro
    
    // #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    int idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                    
                    // Récupérer les coordonnées du maximum
                    auto maxs = max_indices[idx];
                    int max_h = maxs[0];
                    int max_w =  maxs[1];
                    // Propager le gradient uniquement à la position du maximum
                    grad_input(b, c, max_h, max_w) += grad_output(b, c, oh, ow);
                }
            }
        }
    }
    
    return grad_input;
}


std::pair<int, int> MaxPoolLayer::computeOutputDims(int height, int width) const {
    int out_height = static_cast<int>(std::floor((height - pool_size) / static_cast<float>(stride))) + 1;
    int out_width = static_cast<int>(std::floor((width - pool_size) / static_cast<float>(stride))) + 1;
    return {out_height, out_width};
}