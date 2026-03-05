#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <iostream>
#include <memory>
#include <iomanip>
#include <limits>
#include <type_traits>

// ============================================================================
// Classe de base pour les fonctionnalités communes aux tenseurs
// ============================================================================

/**
 * Classe abstraite contenant les utilitaires d'affichage et de statistiques
 * Partagée par Tensor et Tensor3D pour éviter la duplication de code
 */
class TensorUtils {
protected:
    // Structure pour stocker les statistiques d'un tenseur
    struct Statistics {
        float min_val;
        float max_val;
        float mean;
        float l2_norm;
        size_t size;
        
        Statistics() : min_val(std::numeric_limits<float>::max()),
                      max_val(std::numeric_limits<float>::lowest()),
                      mean(0.0f), l2_norm(0.0f), size(0) {}
    };
    
    // Structure pour les paramètres d'affichage
    struct PrintConfig {
        int max_batch = 1;
        int max_channel = -1;
        int max_depth = 2;
        int max_height = 10;
        int max_width = 10;
        bool show_values = true;
        int precision = 4;
        bool use_colors = true;
        
        PrintConfig() = default;
        
        PrintConfig withMaxBatch(int b) const { 
            PrintConfig c = *this; c.max_batch = b; return c; 
        }
        PrintConfig withMaxChannel(int c) const { 
            PrintConfig cfg = *this; cfg.max_channel = c; return cfg; 
        }
        PrintConfig withMaxDepth(int d) const { 
            PrintConfig cfg = *this; cfg.max_depth = d; return cfg; 
        }
        PrintConfig withMaxHeight(int h) const { 
            PrintConfig cfg = *this; cfg.max_height = h; return cfg; 
        }
        PrintConfig withMaxWidth(int w) const { 
            PrintConfig cfg = *this; cfg.max_width = w; return cfg; 
        }
        PrintConfig withShowValues(bool show) const { 
            PrintConfig cfg = *this; cfg.show_values = show; return cfg; 
        }
        PrintConfig withPrecision(int p) const { 
            PrintConfig cfg = *this; cfg.precision = p; return cfg; 
        }
        PrintConfig withColors(bool colors) const { 
            PrintConfig cfg = *this; cfg.use_colors = colors; return cfg; 
        }
    };
    
    // Calcule les statistiques sur un range de valeurs
    template<typename Iterator>
    static Statistics computeStatistics(Iterator begin, Iterator end, size_t total_size) {
        Statistics stats;
        stats.size = total_size;
        
        if (total_size == 0) return stats;
        
        float sum = 0.0f;
        float sum_squares = 0.0f;
        
        for (auto it = begin; it != end; ++it) {
            float val = static_cast<float>(*it);
            stats.min_val = std::min(stats.min_val, val);
            stats.max_val = std::max(stats.max_val, val);
            sum += val;
            sum_squares += val * val;
        }
        
        stats.mean = sum / total_size;
        stats.l2_norm = std::sqrt(sum_squares);
        
        return stats;
    }
    
    // Affiche une ligne de séparation
    static void printSeparator(char c = '=', int length = 63) {
        std::cout << std::string(length, c) << std::endl;
    }
    
    // Affiche une valeur avec couleur optionnelle
    static void printColoredValue(float val, bool use_colors, int precision, int width = 8) {
        if (use_colors) {
            if (val > 0) std::cout << "\033[32m";  // Vert pour positif
            else if (val < 0) std::cout << "\033[31m";  // Rouge pour négatif
            else std::cout << "\033[37m";  // Blanc pour zéro
        }
        
        std::cout << std::fixed << std::setprecision(precision) 
                  << std::setw(width) << val;
        
        if (use_colors) std::cout << "\033[0m";
    }
    
    // Affiche l'en-tête d'un tenseur
    static void printHeader(const std::string& title, const std::vector<int>& shape) {
        printSeparator();
        std::cout << "📊 " << title << ": ";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << "x";
        }
        std::cout << std::endl;
        printSeparator();
    }
    
    // Affiche les statistiques globales
    static void printGlobalStats(const Statistics& stats) {
        std::cout << "📈 Statistiques globales:" << std::endl;
        std::cout << "   Min: " << std::fixed << std::setprecision(4) << stats.min_val << std::endl;
        std::cout << "   Max: " << stats.max_val << std::endl;
        std::cout << "   Mean: " << stats.mean << std::endl;
        std::cout << "   Norm L2: " << stats.l2_norm << std::endl;
        std::cout << "───────────────────────────────────────────────────────────────" << std::endl;
    }
    
    // Affiche le pied de page avec les éléments tronqués
    static void printTruncationNotice(int batch_left, int channel_left, int depth_left) {
        if (batch_left > 0) {
            std::cout << "🔄 ... et " << batch_left << " batch(es) de plus" << std::endl;
        }
        if (channel_left > 0) {
            std::cout << "📌 ... et " << channel_left << " canal(aux) de plus" << std::endl;
        }
        if (depth_left > 0) {
            std::cout << "🧊 ... et " << depth_left << " slice(s) de plus" << std::endl;
        }
        printSeparator();
    }
};

// ============================================================================
// Classe Tensor (4D)
// ============================================================================

class Tensor : public TensorUtils {
private:
    using EigenTensor = Eigen::Tensor<float, 4, Eigen::RowMajor>;
    EigenTensor data;

public:
    // Constructeurs
    Tensor() = default;
    Tensor(int batch, int channels, int height, int width) 
        : data(batch, channels, height, width) {}
    Tensor(const std::vector<int>& shape) 
        : data(shape[0], shape[1], shape[2], shape[3]) {
        if (shape.size() != 4) {
            throw std::runtime_error("Tensor shape must have 4 dimensions");
        }
    }

    // Constructeur par copie
    Tensor(const Tensor& other) : data(other.data.dimensions()) {
        data = other.data;
    }

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            if (data.dimensions() != other.data.dimensions()) {
                data.resize(other.data.dimensions());
            }
            data = other.data;
        }
        return *this;
    }

    // Constructeur de déplacement
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(Tensor&& other) noexcept = default;
    ~Tensor() = default;

    // Accès aux données
    float& operator()(int b, int c, int h, int w) { 
        return data(b, c, h, w); 
    }
    const float& operator()(int b, int c, int h, int w) const { 
        return data(b, c, h, w); 
    }

    float& operator[](size_t index) { return data.data()[index]; }
    const float& operator[](size_t index) const { return data.data()[index]; }

    // Conversion vers Eigen::Matrix
    Eigen::MatrixXf toMatrix() const {
        int batch_size = dim(0);
        int flattened_size = 1;
        for (int i = 1; i < 4; ++i) flattened_size *= dim(i);

        Eigen::Map<const Eigen::MatrixXf> tensor_map(
            getData(), flattened_size, batch_size
        );
        return tensor_map.transpose();
    }

    Eigen::Map<Eigen::MatrixXf> toMatrix(int rows, int cols) {
        if (rows * cols != size()) {
            throw std::runtime_error("Rows * cols must equal tensor size");
        }
        return Eigen::Map<Eigen::MatrixXf>(data.data(), rows, cols);
    }

    Eigen::Map<const Eigen::MatrixXf> toMatrix(int rows, int cols) const {
        if (rows * cols != size()) {
            throw std::runtime_error("Rows * cols must equal tensor size");
        }
        return Eigen::Map<const Eigen::MatrixXf>(data.data(), rows, cols);
    }

    // Conversion depuis Eigen::Matrix
    static Tensor fromMatrix(const Eigen::MatrixXf& matrix, const std::vector<int>& shape) {
        if (shape.size() != 4) {
            throw std::runtime_error("Shape must have 4 dimensions");
        }

        int batch_size = shape[0];
        int flattened_size = 1;
        for (int i = 1; i < 4; ++i) flattened_size *= shape[i];

        if (matrix.rows() != batch_size || matrix.cols() != flattened_size) {
            throw std::runtime_error("Matrix dimensions don't match shape");
        }

        Tensor tensor(shape);
        Eigen::Map<Eigen::MatrixXf> tensor_map(
            tensor.getData(), flattened_size, batch_size
        );
        tensor_map = matrix.transpose();
        return tensor;
    }

    static Tensor fromMatrix(const Eigen::MatrixXf& matrix, int d0, int d1, int d2, int d3) {
        return fromMatrix(matrix, {d0, d1, d2, d3});
    }

    // Reshape
    Tensor reshape(const std::vector<int>& newShape) const {
        if (newShape.size() != 4) {
            throw std::runtime_error("New shape must have 4 dimensions");
        }
        Tensor reshaped;
        reshaped.data = data.reshape(Eigen::array<Eigen::Index, 4>{
            newShape[0], newShape[1], newShape[2], newShape[3]
        });
        return reshaped;
    }

    // Métadonnées
    int dim(int index) const { return static_cast<int>(data.dimension(index)); }
    
    std::vector<int> shape() const {
        return {dim(0), dim(1), dim(2), dim(3)};
    }
    
    int size() const { return static_cast<int>(data.size()); }
    
    void setRandom() { data.setRandom(); }
    void setZero() { data.setZero(); }
    void setConstant(float value) { data.setConstant(value); }

    // Getters Eigen
    EigenTensor& eigen() { return data; }
    const EigenTensor& eigen() const { return data; }

    float* getData() { return data.data(); }
    const float* getData() const { return data.data(); }

    // Affichage
    void printShape() const {
        std::cout << "Tensor shape: (" 
                  << dim(0) << ", " << dim(1) << ", " 
                  << dim(2) << ", " << dim(3) << ")" << std::endl;
    }

    void printByChannel(const PrintConfig& config = PrintConfig()) const {
        if (size() == 0) {
            std::cout << "⚠️  Tenseur vide !" << std::endl;
            return;
        }

        // Statistiques globales
        Statistics global_stats = computeStatistics(
            data.data(), data.data() + size(), size()
        );
        
        printHeader("TENSEUR", shape());
        printGlobalStats(global_stats);

        int batch_size = dim(0);
        int channels = dim(1);
        int height = dim(2);
        int width = dim(3);

        int display_batch = std::min(config.max_batch, batch_size);
        int display_channels = (config.max_channel == -1) ? channels : 
                                std::min(config.max_channel, channels);

        for (int b = 0; b < display_batch; ++b) {
            if (batch_size > 1) {
                std::cout << "🔄 BATCH " << b << " / " << batch_size - 1 << std::endl;
                std::cout << "───────────────────────────────────────────────────────────────" << std::endl;
            }

            for (int c = 0; c < display_channels; ++c) {
                std::cout << "📌 CANAL " << c;
                if (channels > display_channels) {
                    std::cout << " (premier " << display_channels << "/" << channels << ")";
                }
                std::cout << std::endl;

                // Statistiques du canal
                float ch_min = std::numeric_limits<float>::max();
                float ch_max = std::numeric_limits<float>::lowest();
                float ch_sum = 0.0f;

                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        float val = (*this)(b, c, h, w);
                        ch_min = std::min(ch_min, val);
                        ch_max = std::max(ch_max, val);
                        ch_sum += val;
                    }
                }
                float ch_mean = ch_sum / (height * width);

                std::cout << "   Min: " << std::setprecision(config.precision) << ch_min;
                std::cout << " | Max: " << ch_max;
                std::cout << " | Mean: " << ch_mean << std::endl;

                if (config.show_values) {
                    int display_height = std::min(config.max_height, height);
                    int display_width = std::min(config.max_width, width);

                    for (int h = 0; h < display_height; ++h) {
                        std::cout << "   ";
                        for (int w = 0; w < display_width; ++w) {
                            float val = (*this)(b, c, h, w);
                            printColoredValue(val, config.use_colors, config.precision, 8);
                            std::cout << " ";
                        }
                        if (height > display_height && h == display_height - 1) {
                            std::cout << "... (" << height - display_height << " lignes de plus)";
                        }
                        std::cout << std::endl;
                    }

                    if (width > display_width) {
                        std::cout << "   ... (" << width - display_width << " colonnes de plus)" << std::endl;
                    }
                }
                std::cout << "───────────────────────────────────────────────────────────────" << std::endl;
            }
        }

        printTruncationNotice(
            batch_size - display_batch,
            channels - display_channels,
            0
        );
    }

    void quickPrint(const std::string& name = "") const {
        if (!name.empty()) std::cout << name << " ";
        printShape();

        int h = std::min(5, dim(2));
        int w = std::min(5, dim(3));

        std::cout << "Premier canal (5x5):" << std::endl;
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                std::cout << std::setw(8) << std::setprecision(2) 
                          << (*this)(0, 0, i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    void printChannel(int batch_idx, int channel_idx, int max_h = 10, int max_w = 10) const {
        std::cout << "Canal " << channel_idx << " du batch " << batch_idx << ":" << std::endl;

        int height = std::min(max_h, dim(2));
        int width = std::min(max_w, dim(3));

        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                std::cout << std::setw(8) << std::setprecision(4)
                          << (*this)(batch_idx, channel_idx, h, w) << " ";
            }
            if (dim(2) > max_h && h == height - 1) std::cout << "...";
            std::cout << std::endl;
        }
    }

    // Comparaison de tenseurs
    static void compareTensors(const Tensor& t1, const Tensor& t2,
                               const std::string& name1 = "T1",
                               const std::string& name2 = "T2") {
        printHeader("COMPARAISON: " + name1 + " vs " + name2, {});
        
        t1.printShape();
        t2.printShape();

        if (t1.shape() != t2.shape()) {
            std::cout << "❌ Shapes différentes !" << std::endl;
            return;
        }

        float max_diff = 0.0f;
        float sum_diff = 0.0f;
        int n_diff = 0;

        for (int i = 0; i < t1.size(); ++i) {
            float diff = std::abs(t1[i] - t2[i]);
            max_diff = std::max(max_diff, diff);
            sum_diff += diff;
            if (diff > 1e-5f) n_diff++;
        }

        std::cout << "📊 Différence moyenne: " << sum_diff / t1.size() << std::endl;
        std::cout << "📊 Différence maximale: " << max_diff << std::endl;
        std::cout << "📊 Éléments différents: " << n_diff << "/" << t1.size()
                  << " (" << (100.0f * n_diff / t1.size()) << "%)" << std::endl;
        printSeparator();
    }
};

// ============================================================================
// Classe Tensor3D (5D)
// ============================================================================

class Tensor3D : public TensorUtils {
private:
    using EigenTensor3D = Eigen::Tensor<float, 5, Eigen::RowMajor>;
    EigenTensor3D data;

public:
    // Constructeurs
    Tensor3D() = default;
    Tensor3D(int batch, int channels, int depth, int height, int width)
        : data(batch, channels, depth, height, width) {}
    Tensor3D(const std::vector<int>& shape)
        : data(shape[0], shape[1], shape[2], shape[3], shape[4]) {
        if (shape.size() != 5) {
            throw std::runtime_error("Tensor3D shape must have 5 dimensions");
        }
    }

    // Constructeur par copie
    Tensor3D(const Tensor3D& other) : data(other.data.dimensions()) {
        data = other.data;
    }

    Tensor3D& operator=(const Tensor3D& other) {
        if (this != &other) {
            if (data.dimensions() != other.data.dimensions()) {
                data.resize(other.data.dimensions());
            }
            data = other.data;
        }
        return *this;
    }

    // Constructeur de déplacement
    Tensor3D(Tensor3D&& other) noexcept = default;
    Tensor3D& operator=(Tensor3D&& other) noexcept = default;
    ~Tensor3D() = default;

    // Accès aux données
    float& operator()(int b, int c, int d, int h, int w) {
        return data(b, c, d, h, w);
    }
    const float& operator()(int b, int c, int d, int h, int w) const {
        return data(b, c, d, h, w);
    }

    float& operator[](size_t index) { return data.data()[index]; }
    const float& operator[](size_t index) const { return data.data()[index]; }

    // Conversion vers Eigen::Matrix
    Eigen::MatrixXf toMatrix() const {
        int batch_size = dim(0);
        int flattened_size = 1;
        for (int i = 1; i < 5; ++i) flattened_size *= dim(i);

        Eigen::Map<const Eigen::MatrixXf> tensor_map(
            getData(), flattened_size, batch_size
        );
        return tensor_map.transpose();
    }

    Eigen::Map<Eigen::MatrixXf> toMatrix(int rows, int cols) {
        if (rows * cols != size()) {
            throw std::runtime_error("Rows * cols must equal tensor size");
        }
        return Eigen::Map<Eigen::MatrixXf>(data.data(), rows, cols);
    }

    Eigen::Map<const Eigen::MatrixXf> toMatrix(int rows, int cols) const {
        if (rows * cols != size()) {
            throw std::runtime_error("Rows * cols must equal tensor size");
        }
        return Eigen::Map<const Eigen::MatrixXf>(data.data(), rows, cols);
    }

    // Conversion depuis Eigen::Matrixf
    static Tensor3D fromMatrix(const Eigen::MatrixXf& matrix, const std::vector<int>& shape) {
        if (shape.size() != 5) {
            throw std::runtime_error("Shape must have 5 dimensions");
        }

        int batch_size = shape[0];
        int flattened_size = 1;
        for (int i = 1; i < 5; ++i) flattened_size *= shape[i];

        if (matrix.rows() != batch_size || matrix.cols() != flattened_size) {
            throw std::runtime_error("Matrix dimensions don't match shape");
        }

        Tensor3D tensor(shape);
        Eigen::Map<Eigen::MatrixXf> tensor_map(
            tensor.getData(), flattened_size, batch_size
        );
        tensor_map = matrix.transpose();
        return tensor;
    }

    static Tensor3D fromMatrix(const Eigen::MatrixXf& matrix, 
                               int d0, int d1, int d2, int d3, int d4) {
        return fromMatrix(matrix, {d0, d1, d2, d3, d4});
    }

    // Reshape
    Tensor3D reshape(const std::vector<int>& newShape) const {
        if (newShape.size() != 5) {
            throw std::runtime_error("New shape must have 5 dimensions");
        }
        Tensor3D reshaped;
        reshaped.data = data.reshape(Eigen::array<Eigen::Index, 5>{
            newShape[0], newShape[1], newShape[2], newShape[3], newShape[4]
        });
        return reshaped;
    }

    // Métadonnées
    int dim(int index) const { return static_cast<int>(data.dimension(index)); }
    
    std::vector<int> shape() const {
        return {dim(0), dim(1), dim(2), dim(3), dim(4)};
    }
    
    int size() const { return static_cast<int>(data.size()); }
    
    void setRandom() { data.setRandom(); }
    void setZero() { data.setZero(); }
    void setConstant(float value) { data.setConstant(value); }

    // Getters Eigen
    EigenTensor3D& eigen() { return data; }
    const EigenTensor3D& eigen() const { return data; }

    float* getData() { return data.data(); }
    const float* getData() const { return data.data(); }

    // Affichage
    void printShape() const {
        std::cout << "Tensor3D shape: (" 
                  << dim(0) << ", " << dim(1) << ", " << dim(2) << ", "
                  << dim(3) << ", " << dim(4) << ")" << std::endl;
    }

    void printByChannel(const PrintConfig& config = PrintConfig()) const {
        if (size() == 0) {
            std::cout << "⚠️  Tenseur vide !" << std::endl;
            return;
        }

        // Statistiques globales
        Statistics global_stats = computeStatistics(
            data.data(), data.data() + size(), size()
        );
        
        printHeader("TENSEUR 3D", shape());
        printGlobalStats(global_stats);

        int batch_size = dim(0);
        int channels = dim(1);
        int depth = dim(2);
        int height = dim(3);
        int width = dim(4);

        int display_batch = std::min(config.max_batch, batch_size);
        int display_channels = (config.max_channel == -1) ? channels : 
                                std::min(config.max_channel, channels);
        int display_depth = std::min(config.max_depth, depth);
        int display_height = std::min(config.max_height, height);
        int display_width = std::min(config.max_width, width);

        for (int b = 0; b < display_batch; ++b) {
            if (batch_size > 1) {
                std::cout << "🔄 BATCH " << b << " / " << batch_size - 1 << std::endl;
                std::cout << "───────────────────────────────────────────────────────────────" << std::endl;
            }

            for (int c = 0; c < display_channels; ++c) {
                std::cout << "📌 CANAL " << c;
                if (channels > display_channels) {
                    std::cout << " (premier " << display_channels << "/" << channels << ")";
                }
                std::cout << std::endl;

                // Statistiques du canal
                float ch_min = std::numeric_limits<float>::max();
                float ch_max = std::numeric_limits<float>::lowest();
                float ch_sum = 0.0f;

                for (int d = 0; d < depth; ++d) {
                    for (int h = 0; h < height; ++h) {
                        for (int w = 0; w < width; ++w) {
                            float val = (*this)(b, c, d, h, w);
                            ch_min = std::min(ch_min, val);
                            ch_max = std::max(ch_max, val);
                            ch_sum += val;
                        }
                    }
                }
                float ch_mean = ch_sum / (depth * height * width);

                std::cout << "   Min: " << std::setprecision(config.precision) << ch_min;
                std::cout << " | Max: " << ch_max;
                std::cout << " | Mean: " << ch_mean << std::endl;
                std::cout << "───────────────────────────────────────────────────────────────" << std::endl;

                if (config.show_values) {
                    for (int d = 0; d < display_depth; ++d) {
                        std::cout << "  🧊 Profondeur (slice) " << d << "/" << depth - 1 << ":" << std::endl;

                        for (int h = 0; h < display_height; ++h) {
                            std::cout << "   ";
                            for (int w = 0; w < display_width; ++w) {
                                float val = (*this)(b, c, d, h, w);
                                printColoredValue(val, config.use_colors, config.precision, 8);
                                std::cout << " ";
                            }
                            if (width > display_width && h == display_height - 1) {
                                std::cout << "... (" << width - display_width << " col. de plus)";
                            }
                            std::cout << std::endl;
                        }

                        if (height > display_height) {
                            std::cout << "   ... (" << height - display_height << " lignes de plus)" << std::endl;
                        }
                    }

                    if (depth > display_depth) {
                        std::cout << "  🧊 ... et " << depth - display_depth << " slice(s) de plus" << std::endl;
                    }
                }
                std::cout << "───────────────────────────────────────────────────────────────" << std::endl;
            }
        }

        printTruncationNotice(
            batch_size - display_batch,
            channels - display_channels,
            depth - display_depth
        );
    }

    void quickPrint(const std::string& name = "") const {
        if (!name.empty()) std::cout << name << " ";
        printShape();

        int h = std::min(5, dim(3));
        int w = std::min(5, dim(4));

        std::cout << "Première slice du canal 0 (5x5):" << std::endl;
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                std::cout << std::setw(8) << std::setprecision(2)
                          << (*this)(0, 0, 0, i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    void printSlice(int batch_idx, int channel_idx, int depth_idx,
                    int max_h = 10, int max_w = 10) const {
        std::cout << "Canal " << channel_idx
                  << ", slice " << depth_idx
                  << ", batch " << batch_idx << ":" << std::endl;

        int height = std::min(max_h, dim(3));
        int width = std::min(max_w, dim(4));

        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                std::cout << std::setw(8) << std::setprecision(4)
                          << (*this)(batch_idx, channel_idx, depth_idx, h, w) << " ";
            }
            if (dim(3) > max_h && h == height - 1) std::cout << "...";
            std::cout << std::endl;
        }
    }

    // Comparaison de tenseurs 3D
    static void compareTensor3Ds(const Tensor3D& t1, const Tensor3D& t2,
                                 const std::string& name1 = "T1",
                                 const std::string& name2 = "T2") {
        printHeader("COMPARAISON 3D: " + name1 + " vs " + name2, {});
        
        t1.printShape();
        t2.printShape();

        if (t1.shape() != t2.shape()) {
            std::cout << "❌ Shapes différentes !" << std::endl;
            return;
        }

        float max_diff = 0.0f;
        float sum_diff = 0.0f;
        int n_diff = 0;

        for (int i = 0; i < t1.size(); ++i) {
            float diff = std::abs(t1[i] - t2[i]);
            max_diff = std::max(max_diff, diff);
            sum_diff += diff;
            if (diff > 1e-5f) n_diff++;
        }

        std::cout << "📊 Différence moyenne: " << sum_diff / t1.size() << std::endl;
        std::cout << "📊 Différence maximale: " << max_diff << std::endl;
        std::cout << "📊 Éléments différents: " << n_diff << "/" << t1.size()
                  << " (" << (100.0f * n_diff / t1.size()) << "%)" << std::endl;
        printSeparator();
    }
};

// ============================================================================
// Surcharge d'opérateurs pour faciliter l'utilisation
// ============================================================================

inline std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    tensor.quickPrint();
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const Tensor3D& tensor) {
    tensor.quickPrint();
    return os;
}