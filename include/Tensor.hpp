#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// Tensor — classe unifiée 2D/3D
//
// Stockage interne : Eigen::Tensor<float, 5, RowMajor>
//                   dimensions : (d0, d1, d2, d3, d4)
//
// Mode 4D (couches 2D) : (B, C, H, W)  → stocké comme (B, C, 1, H, W)
// Mode 5D (couches 3D) : (B, C, D, H, W) → stocké nativement
//
// L'interface Layer::forward(const Tensor&) fonctionne pour les deux cas
// sans aucune modification de la hiérarchie Layer.
// ─────────────────────────────────────────────────────────────────────────────
class Tensor {
private:
    using EigenTensor = Eigen::Tensor<float, 5, Eigen::RowMajor>;
    EigenTensor data;

    // Rang logique : 4 pour les tenseurs 2D, 5 pour les tenseurs 3D
    // Déterminé à la construction, conservé pour l'affichage et les validations
    int logical_rank = 5;

public:

    // ─────────────────────────────────────────────────────────────────────────
    // Constructeurs
    // ─────────────────────────────────────────────────────────────────────────

    Tensor() : data(0, 0, 1, 0, 0), logical_rank(4) {}

    // ── Mode 4D : (B, C, H, W) → stocké comme (B, C, 1, H, W) ──────────────
    Tensor(int batch, int channels, int height, int width)
        : data(batch, channels, 1, height, width), logical_rank(4) {}

    // ── Mode 5D : (B, C, D, H, W) ───────────────────────────────────────────
    Tensor(int batch, int channels, int depth, int height, int width)
        : data(batch, channels, depth, height, width), logical_rank(5) {}

    // ── Construction depuis shape vector ─────────────────────────────────────
    explicit Tensor(const std::vector<int>& shape) {
        if (shape.size() == 4) {
            data = EigenTensor(shape[0], shape[1], 1, shape[2], shape[3]);
            logical_rank = 4;
        } else if (shape.size() == 5) {
            data = EigenTensor(shape[0], shape[1], shape[2], shape[3], shape[4]);
            logical_rank = 5;
        } else {
            throw std::runtime_error(
                "[Tensor] shape doit avoir 4 ou 5 dimensions, reçu: "
                + std::to_string(shape.size()));
        }
    }

    // ── Copie / déplacement ──────────────────────────────────────────────────
    Tensor(const Tensor& other) : data(other.data.dimensions()), logical_rank(other.logical_rank) {
        data = other.data;
    }

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            if (data.dimensions() != other.data.dimensions())
                data.resize(other.data.dimensions());
            data = other.data;
            logical_rank = other.logical_rank;
        }
        return *this;
    }

    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(Tensor&& other) noexcept = default;
    ~Tensor() = default;

    // ─────────────────────────────────────────────────────────────────────────
    // Accès aux données
    // ─────────────────────────────────────────────────────────────────────────

    // ── Accès 5D natif (couches 3D) ──────────────────────────────────────────
    float& operator()(int b, int c, int d, int h, int w) {
        return data(b, c, d, h, w);
    }
    const float& operator()(int b, int c, int d, int h, int w) const {
        return data(b, c, d, h, w);
    }

    // ── Accès 4D — alias sur d=0 (couches 2D existantes, sans modification) ──
    float& operator()(int b, int c, int h, int w) {
        return data(b, c, 0, h, w);
    }
    const float& operator()(int b, int c, int h, int w) const {
        return data(b, c, 0, h, w);
    }

    // ── Accès plat ───────────────────────────────────────────────────────────
    float& operator[](size_t index) { return data.data()[index]; }
    const float& operator[](size_t index) const { return data.data()[index]; }

    // ─────────────────────────────────────────────────────────────────────────
    // Métadonnées
    // ─────────────────────────────────────────────────────────────────────────

    // ndim() : rang logique (4 ou 5)
    int ndim() const { return logical_rank; }

    // dim(i) : dimension selon le rang logique
    // Mode 4D : dim(0)=B  dim(1)=C  dim(2)=H  dim(3)=W
    // Mode 5D : dim(0)=B  dim(1)=C  dim(2)=D  dim(3)=H  dim(4)=W
    int dim(int index) const {
        if (logical_rank == 4) {
            // Mapping 4D → indices internes 5D
            // dim(0)→d0  dim(1)→d1  dim(2)→d3  dim(3)→d4  (d2=1 ignoré)
            const int map4[4] = {0, 1, 3, 4};
            if (index < 0 || index >= 4)
                throw std::out_of_range("[Tensor] dim index hors limites (mode 4D: 0-3)");
            return static_cast<int>(data.dimension(map4[index]));
        } else {
            if (index < 0 || index >= 5)
                throw std::out_of_range("[Tensor] dim index hors limites (mode 5D: 0-4)");
            return static_cast<int>(data.dimension(index));
        }
    }

    // shape() : vecteur selon le rang logique
    std::vector<int> shape() const {
        if (logical_rank == 4) {
            return { static_cast<int>(data.dimension(0)),
                     static_cast<int>(data.dimension(1)),
                     static_cast<int>(data.dimension(3)),
                     static_cast<int>(data.dimension(4)) };
        } else {
            return { static_cast<int>(data.dimension(0)),
                     static_cast<int>(data.dimension(1)),
                     static_cast<int>(data.dimension(2)),
                     static_cast<int>(data.dimension(3)),
                     static_cast<int>(data.dimension(4)) };
        }
    }

    // Accès direct aux dimensions internes (toujours 5D)
    int dim5(int index) const {
        if (index < 0 || index >= 5)
            throw std::out_of_range("[Tensor] dim5 index hors limites");
        return static_cast<int>(data.dimension(index));
    }

    int size() const { return static_cast<int>(data.size()); }

    // ─────────────────────────────────────────────────────────────────────────
    // Initialisation
    // ─────────────────────────────────────────────────────────────────────────

    void setRandom()               { data.setRandom(); }
    void setZero()                 { data.setZero();   }
    void setConstant(float value)  { data.setConstant(value); }

    // ─────────────────────────────────────────────────────────────────────────
    // Données brutes
    // ─────────────────────────────────────────────────────────────────────────

    float*       getData()       { return data.data(); }
    const float* getData() const { return data.data(); }

    EigenTensor&       eigen()       { return data; }
    const EigenTensor& eigen() const { return data; }

    // ─────────────────────────────────────────────────────────────────────────
    // Conversion Eigen::Matrix ↔ Tensor
    // ─────────────────────────────────────────────────────────────────────────

    // Aplatit toutes les dimensions sauf le batch → (B, C*[D]*H*W)
    Eigen::MatrixXf toMatrix() const {
        int B = static_cast<int>(data.dimension(0));
        int flat = size() / B;
        Eigen::Map<const Eigen::MatrixXf> m(getData(), flat, B);
        return m.transpose();
    }

    Eigen::Map<Eigen::MatrixXf> toMatrix(int rows, int cols) {
        if (rows * cols != size())
            throw std::runtime_error("[Tensor] toMatrix: rows*cols != size");
        return Eigen::Map<Eigen::MatrixXf>(getData(), rows, cols);
    }

    Eigen::Map<const Eigen::MatrixXf> toMatrix(int rows, int cols) const {
        if (rows * cols != size())
            throw std::runtime_error("[Tensor] toMatrix: rows*cols != size");
        return Eigen::Map<const Eigen::MatrixXf>(getData(), rows, cols);
    }

    // ── fromMatrix 4D ────────────────────────────────────────────────────────
    static Tensor fromMatrix(const Eigen::MatrixXf& matrix,
                             const std::vector<int>& shape) {
        Tensor t(shape);
        int B    = shape[0];
        int flat = t.size() / B;
        if (matrix.rows() != B || matrix.cols() != flat)
            throw std::runtime_error("[Tensor] fromMatrix: dimensions incompatibles");
        Eigen::Map<Eigen::MatrixXf> m(t.getData(), flat, B);
        m = matrix.transpose();
        return t;
    }

    // Surcharges pratiques
    static Tensor fromMatrix(const Eigen::MatrixXf& m,
                             int d0, int d1, int d2, int d3) {
        return fromMatrix(m, {d0, d1, d2, d3});
    }
    static Tensor fromMatrix(const Eigen::MatrixXf& m,
                             int d0, int d1, int d2, int d3, int d4) {
        return fromMatrix(m, {d0, d1, d2, d3, d4});
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Reshape
    // ─────────────────────────────────────────────────────────────────────────

    Tensor reshape(const std::vector<int>& newShape) const {
        Tensor reshaped(newShape);
        if (reshaped.size() != size())
            throw std::runtime_error("[Tensor] reshape: taille incompatible");
        // Expression lazy Eigen : aucun buffer intermédiaire, une seule écriture
        reshaped.eigen() = data.reshape(reshaped.eigen().dimensions());
        return reshaped;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Conversion de rang logique
    // ─────────────────────────────────────────────────────────────────────────

    // Réinterprète un Tensor 4D comme 5D avec D=1
    // Utile à la frontière 2D → 3D
    Tensor as5D() const {
        if (logical_rank == 5) return *this;
        Tensor t5;
        t5.data = data;
        t5.logical_rank = 5;
        return t5;
    }

    // Réinterprète un Tensor 5D avec D=1 comme 4D
    Tensor as4D() const {
        if (logical_rank == 4) return *this;
        if (static_cast<int>(data.dimension(2)) != 1)
            throw std::runtime_error(
                "[Tensor] as4D: dimension D=" +
                std::to_string(data.dimension(2)) + " != 1");
        Tensor t4;
        t4.data = data;
        t4.logical_rank = 4;
        return t4;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Affichage
    // ─────────────────────────────────────────────────────────────────────────

    void printShape() const {
        std::cout << "Tensor shape: (";
        auto s = shape();
        for (size_t i = 0; i < s.size(); ++i) {
            std::cout << s[i];
            if (i + 1 < s.size()) std::cout << ", ";
        }
        std::cout << ")  [" << logical_rank << "D]" << std::endl;
    }

    // Affichage détaillé — adapté automatiquement au rang logique
    void printByChannel(int max_batch   = 1,
                        int max_channel = -1,
                        int max_depth   = 2,   // ignoré en mode 4D
                        int max_height  = 10,
                        int max_width   = 10,
                        bool show_values = true,
                        int  precision   = 4) const {

        if (size() == 0) { std::cout << "⚠️  Tenseur vide !" << std::endl; return; }

        // Dimensions selon le rang logique
        int B = dim(0);
        int C = dim(1);
        int D = (logical_rank == 5) ? dim(2) : 1;
        int H = (logical_rank == 5) ? dim(3) : dim(2);
        int W = (logical_rank == 5) ? dim(4) : dim(3);

        int dB = std::min(max_batch,   B);
        int dC = (max_channel == -1) ? C : std::min(max_channel, C);
        int dD = std::min(max_depth,   D);
        int dH = std::min(max_height,  H);
        int dW = std::min(max_width,   W);

        // En-tête
        std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
        if (logical_rank == 4) {
            std::cout << "📊 TENSEUR 4D: "
                      << B << "x" << C << "x" << H << "x" << W << std::endl;
        } else {
            std::cout << "📊 TENSEUR 5D: "
                      << B << "x" << C << "x" << D << "x" << H << "x" << W << std::endl;
        }
        std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;

        // Stats globales
        float gmin = std::numeric_limits<float>::max();
        float gmax = std::numeric_limits<float>::lowest();
        float gsum = 0.f, gsumsq = 0.f;
        for (int i = 0; i < size(); ++i) {
            float v = (*this)[i];
            gmin = std::min(gmin, v);
            gmax = std::max(gmax, v);
            gsum += v;
            gsumsq += v * v;
        }
        std::cout << "📈 Statistiques globales:" << std::endl;
        std::cout << std::fixed << std::setprecision(precision);
        std::cout << "   Min: "    << gmin << std::endl;
        std::cout << "   Max: "    << gmax << std::endl;
        std::cout << "   Mean: "   << gsum / size() << std::endl;
        std::cout << "   Norm L2: "<< std::sqrt(gsumsq) << std::endl;
        std::cout << "───────────────────────────────────────────────────────────────" << std::endl;

        for (int b = 0; b < dB; ++b) {
            if (B > 1) {
                std::cout << "🔄 BATCH " << b << " / " << B - 1 << std::endl;
                std::cout << "───────────────────────────────────────────────────────────────" << std::endl;
            }

            for (int c = 0; c < dC; ++c) {
                std::cout << "📌 CANAL " << c;
                if (C > dC) std::cout << " (" << dC << "/" << C << ")";
                std::cout << std::endl;

                // Stats du canal
                float cmin = std::numeric_limits<float>::max();
                float cmax = std::numeric_limits<float>::lowest();
                float csum = 0.f;
                for (int d = 0; d < D; ++d)
                for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w) {
                    float v = (logical_rank == 5)
                              ? (*this)(b, c, d, h, w)
                              : (*this)(b, c, h, w);
                    cmin = std::min(cmin, v);
                    cmax = std::max(cmax, v);
                    csum += v;
                }
                std::cout << "   Min: " << cmin
                          << " | Max: " << cmax
                          << " | Mean: " << csum / (D * H * W) << std::endl;
                std::cout << "───────────────────────────────────────────────────────────────" << std::endl;

                if (!show_values) continue;

                // Affichage des valeurs
                for (int d = 0; d < dD; ++d) {
                    if (logical_rank == 5)
                        std::cout << "  🧊 Slice " << d << "/" << D - 1 << ":" << std::endl;

                    for (int h = 0; h < dH; ++h) {
                        std::cout << "   ";
                        for (int w = 0; w < dW; ++w) {
                            float v = (logical_rank == 5)
                                      ? (*this)(b, c, d, h, w)
                                      : (*this)(b, c, h, w);
                            if      (v > 0) std::cout << "\033[32m";
                            else if (v < 0) std::cout << "\033[31m";
                            else            std::cout << "\033[37m";
                            std::cout << std::setw(8) << v << "\033[0m ";
                        }
                        if (W > dW) std::cout << "... (" << W - dW << " col. de plus)";
                        std::cout << std::endl;
                    }
                    if (H > dH)
                        std::cout << "   ... (" << H - dH << " lignes de plus)" << std::endl;
                }
                if (logical_rank == 5 && D > dD)
                    std::cout << "  🧊 ... et " << D - dD << " slice(s) de plus" << std::endl;

                std::cout << "───────────────────────────────────────────────────────────────" << std::endl;
            }

            if (C > dC)
                std::cout << "📌 ... et " << C - dC << " canal(aux) de plus" << std::endl;
        }

        if (B > dB)
            std::cout << "🔄 ... et " << B - dB << " batch(es) de plus" << std::endl;
        std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
    }

    // Affichage rapide
    void quickPrint(const std::string& name = "") const {
        if (!name.empty()) std::cout << name << " ";
        printShape();

        if (logical_rank == 4) {
            int h = std::min(5, dim(2));
            int w = std::min(5, dim(3));
            std::cout << "Premier canal (max 5x5):" << std::endl;
            for (int i = 0; i < h; ++i) {
                for (int j = 0; j < w; ++j)
                    std::cout << std::setw(8) << std::setprecision(2)
                              << (*this)(0, 0, i, j) << " ";
                std::cout << std::endl;
            }
        } else {
            int h = std::min(5, dim(3));
            int w = std::min(5, dim(4));
            std::cout << "Première slice du canal 0 (max 5x5):" << std::endl;
            for (int i = 0; i < h; ++i) {
                for (int j = 0; j < w; ++j)
                    std::cout << std::setw(8) << std::setprecision(2)
                              << (*this)(0, 0, 0, i, j) << " ";
                std::cout << std::endl;
            }
        }
    }

    // Affiche un canal 2D (mode 4D) ou une slice (mode 5D)
    void printChannel(int batch_idx, int channel_idx,
                      int depth_idx = 0,
                      int max_h = 10, int max_w = 10) const {
        if (logical_rank == 4) {
            std::cout << "Canal " << channel_idx << ", batch " << batch_idx << ":" << std::endl;
            int H = std::min(max_h, dim(2));
            int W = std::min(max_w, dim(3));
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w)
                    std::cout << std::setw(8) << std::setprecision(4)
                              << (*this)(batch_idx, channel_idx, h, w) << " ";
                std::cout << std::endl;
            }
        } else {
            std::cout << "Canal " << channel_idx
                      << ", slice " << depth_idx
                      << ", batch " << batch_idx << ":" << std::endl;
            int H = std::min(max_h, dim(3));
            int W = std::min(max_w, dim(4));
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w)
                    std::cout << std::setw(8) << std::setprecision(4)
                              << (*this)(batch_idx, channel_idx, depth_idx, h, w) << " ";
                std::cout << std::endl;
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Comparaison
    // ─────────────────────────────────────────────────────────────────────────

    static void compare(const Tensor& t1, const Tensor& t2,
                        const std::string& name1 = "T1",
                        const std::string& name2 = "T2") {
        std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
        std::cout << "🔍 COMPARAISON: " << name1 << " vs " << name2 << std::endl;
        std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
        t1.printShape();
        t2.printShape();

        if (t1.shape() != t2.shape()) {
            std::cout << "❌ Shapes différentes !" << std::endl;
            return;
        }

        float max_diff = 0.f, sum_diff = 0.f;
        int   n_diff   = 0;
        for (int i = 0; i < t1.size(); ++i) {
            float d = std::abs(t1[i] - t2[i]);
            max_diff  = std::max(max_diff, d);
            sum_diff += d;
            if (d > 1e-5f) n_diff++;
        }
        std::cout << "📊 Différence moyenne:  " << sum_diff / t1.size() << std::endl;
        std::cout << "📊 Différence maximale: " << max_diff << std::endl;
        std::cout << "📊 Éléments différents: " << n_diff << "/" << t1.size()
                  << " (" << 100.f * n_diff / t1.size() << "%)" << std::endl;
        std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
    }

    // Alias pour compatibilité avec votre ancien code
    void compareTensors(const Tensor& t1, const Tensor& t2,
                        const std::string& n1 = "T1",
                        const std::string& n2 = "T2") {
        compare(t1, t2, n1, n2);
    }


// Version inefficace (avec copie)
Tensor getBatch_i_copy(int i) const {
    Tensor batch(this->dim(1), this->dim(2), this->dim(3), this->dim(4));
    
    for (int c = 0; c < dim(1); ++c)
        for (int d = 0; d < dim(2); ++d)
            for (int h = 0; h < dim(3); ++h)
                for (int w = 0; w < dim(4); ++w)
                    batch(c, d, h, w) = (*this)(i, c, d, h, w);
    
    return batch;  // Copie coûteuse !
}
};

// ─────────────────────────────────────────────────────────────────────────────
// Surcharge stream
// ─────────────────────────────────────────────────────────────────────────────
inline std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    t.quickPrint();
    return os;
}