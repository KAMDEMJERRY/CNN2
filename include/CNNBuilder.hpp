#pragma once
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <stdexcept>
#include "Tensor.hpp"

class CNNBuilder {
private:
    // Format: {height, width, channels}
    std::vector<int> current_shape = {224, 224, 3};
    std::vector<std::pair<std::string, std::vector<int>>> layer_history;
    int flatten_size = 0;
public:
    // Constructeur
    CNNBuilder(int H, int W, int C) {
        if (H <= 0 || W <= 0 || C <= 0) {
            throw std::invalid_argument("Dimensions must be positive");
        }
        current_shape = {H, W, C};
        layer_history.push_back({"Input", current_shape});
    }
    
    CNNBuilder(const Tensor& input) {
        current_shape = {input.dim(2), input.dim(3), input.dim(1)};
        layer_history.push_back({"Input", current_shape});
    }
    
    // ==================== COUCHES CONVOLUTIVES ====================
    
    // Version COMPLÈTE avec tous les paramètres
    void addConv(int out_channels, 
                 int kernel_h, int kernel_w,
                 int stride_h, int stride_w,
                 int pad_h, int pad_w,
                 int dilation_h = 1, int dilation_w = 1) {
        
        int in_h = current_shape[0];
        int in_w = current_shape[1];
        int in_c = current_shape[2];
        
        // Vérifications
        if (kernel_h <= 0 || kernel_w <= 0 || stride_h <= 0 || stride_w <= 0) {
            throw std::invalid_argument("Kernel and stride must be positive");
        }
        
        // Formule avec dilation
        int effective_kernel_h = kernel_h + (kernel_h - 1) * (dilation_h - 1);
        int effective_kernel_w = kernel_w + (kernel_w - 1) * (dilation_w - 1);
        
        int out_h = (in_h + 2 * pad_h - effective_kernel_h) / stride_h + 1;
        int out_w = (in_w + 2 * pad_w - effective_kernel_w) / stride_w + 1;
        
        if (out_h <= 0 || out_w <= 0) {
            throw std::runtime_error("Invalid output dimensions: " + 
                                    std::to_string(out_h) + "x" + std::to_string(out_w));
        }
        
        std::cout << "Conv: " << in_c << "→" << out_channels 
                  << ", " << in_h << "x" << in_w 
                  << " → " << out_h << "x" << out_w
                  << " (kernel=" << kernel_h << "x" << kernel_w
                  << ", stride=" << stride_h << "x" << stride_w
                  << ", pad=" << pad_h << "x" << pad_w
                  << ", dilation=" << dilation_h << "x" << dilation_w << ")"
                  << std::endl;
        
        current_shape = {out_h, out_w, out_channels};
        layer_history.push_back({"Conv", current_shape});
    }
    
    // Surcharge pour kernel carré (paramètres par défaut)
    void addConv(int out_channels, int kernel_size,
                 int stride = 1, int padding = 0, int dilation = 1) {
        addConv(out_channels, 
                kernel_size, kernel_size,
                stride, stride,
                padding, padding,
                dilation, dilation);
    }
    
    // Padding "same" (préserve la taille)
    void addConvSame(int out_channels, int kernel_size, int stride = 1) {
        int padding = (kernel_size - 1) / 2;
        // S'assurer que la taille est préservée
        if ((current_shape[0] + 2*padding - kernel_size) % stride != 0) {
            std::cerr << "Warning: 'same' padding may not preserve exact size" << std::endl;
        }
        addConv(out_channels, kernel_size, stride, padding, 1);
    }
    
    // ==================== COUCHES DE POOLING ====================
    
    // Version COMPLÈTE - RENOMMÉE pour éviter l'ambiguïté
    void addMaxPoolFull(int pool_h, int pool_w,
                        int stride_h, int stride_w,
                        int pad_h = 0, int pad_w = 0) {
        
        int in_h = current_shape[0];
        int in_w = current_shape[1];
        int in_c = current_shape[2];
        
        if (pool_h <= 0 || pool_w <= 0 || stride_h <= 0 || stride_w <= 0) {
            throw std::invalid_argument("Pool and stride must be positive");
        }
        
        int out_h = (in_h + 2 * pad_h - pool_h) / stride_h + 1;
        int out_w = (in_w + 2 * pad_w - pool_w) / stride_w + 1;
        
        if (out_h <= 0 || out_w <= 0) {
            throw std::runtime_error("Invalid pool output dimensions");
        }
        
        std::cout << "MaxPool: " << in_h << "x" << in_w 
                  << " → " << out_h << "x" << out_w
                  << " (pool=" << pool_h << "x" << pool_w
                  << ", stride=" << stride_h << "x" << stride_w
                  << ", pad=" << pad_h << "x" << pad_w << ")"
                  << std::endl;
        
        current_shape = {out_h, out_w, in_c};
        layer_history.push_back({"MaxPool", current_shape});
    }
    
    // Version simplifiée - pooling carré
    void addMaxPool(int pool_size, int stride = -1, int padding = 0) {
        if (stride == -1) stride = pool_size;
        addMaxPoolFull(pool_size, pool_size, stride, stride, padding, padding);
    }
    
    // Version pour pooling rectangle (2 paramètres: hauteur et largeur)
    void addMaxPool(int pool_h, int pool_w) {
        addMaxPoolFull(pool_h, pool_w, pool_h, pool_w, 0, 0);
    }
    
    // Version avec stride différent
    void addMaxPool(int pool_h, int pool_w, int stride_h, int stride_w) {
        addMaxPoolFull(pool_h, pool_w, stride_h, stride_w, 0, 0);
    }
    
    // AveragePool
    void addAvgPool(int pool_size, int stride = -1, int padding = 0) {
        if (stride == -1) stride = pool_size;
        
        int in_h = current_shape[0];
        int in_w = current_shape[1];
        int in_c = current_shape[2];
        
        int out_h = (in_h + 2 * padding - pool_size) / stride + 1;
        int out_w = (in_w + 2 * padding - pool_size) / stride + 1;
        
        std::cout << "AvgPool: " << in_h << "x" << in_w 
                  << " → " << out_h << "x" << out_w << std::endl;
        
        current_shape = {out_h, out_w, in_c};
        layer_history.push_back({"AvgPool", current_shape});
    }
    
    // Global Pooling
    void addGlobalAvgPool() {
        current_shape = {1, 1, current_shape[2]};
        std::cout << "GlobalAvgPool: → 1x1x" << current_shape[2] << std::endl;
        layer_history.push_back({"GlobalAvgPool", current_shape});
    }
    
    // ==================== COUCHES DENSES ====================
    
    int getFlattenSize() const {
        return current_shape[0] * current_shape[1] * current_shape[2];
    }
    
    void addFlatten() {
        int size = getFlattenSize();
        flatten_size = size;
        std::cout << "Flatten: " 
                  << current_shape[2] << "x" << current_shape[0] << "x" << current_shape[1]
                  << " → " << size << std::endl;
        current_shape = {1, 1, size};
        layer_history.push_back({"Flatten", current_shape});
    }
    
    void addDense(int output_size) {
        int input_size = current_shape[2];
        std::cout << "Dense: " << input_size << " → " << output_size << std::endl;
        current_shape = {1, 1, output_size};
        layer_history.push_back({"Dense", current_shape});
    }
    
    // ==================== UTILITAIRES ====================
    
    std::vector<int> getCurrentShape() const {
        return current_shape;
    }
    
    void printCurrentShape() const {
        std::cout << "▶ Current shape: " 
                  << current_shape[2] << " × " 
                  << current_shape[0] << " × " 
                  << current_shape[1] 
                  << " (C×H×W)" << std::endl;
    }
    
    void printArchitecture() const {
        std::cout << "\n═══════════════════════════════════════════" << std::endl;
        std::cout << "        ARCHITECTURE CNN" << std::endl;
        std::cout << "═══════════════════════════════════════════" << std::endl;
        
        for (const auto& [name, shape] : layer_history) {
            std::cout << std::left << std::setw(15) << name << ": ";
            if (name == "Flatten" || name == "Dense") {
                std::cout << shape[2];
            } else {
                std::cout << shape[2] << "x" << shape[0] << "x" << shape[1];
            }
            std::cout << std::endl;
        }
        std::cout << "═══════════════════════════════════════════" << std::endl;
    }

    int getFirstDenseInputSize(){
        return flatten_size;
    }
};