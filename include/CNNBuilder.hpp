#pragma once
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <stdexcept>
#include "Tensor.hpp"

// =============================================================================
// CNNBuilder — calculateur de dimensions pour architectures 2D et 3D
//
// Mode 2D : forme interne {C, H, W}        — construit depuis Tensor 4D
// Mode 3D : forme interne {C, D, H, W}     — construit depuis Tensor 5D
//
// Les méthodes addConv / addMaxPool etc. sont communes aux deux modes via
// des surcharges. Les méthodes *3D requièrent le mode 3D (exception sinon).
// =============================================================================
class CNNBuilder {
public:

    // =========================================================================
    // Constructeurs
    // =========================================================================

    // 2D explicite
    CNNBuilder(int H, int W, int C) {
        requirePositive({H, W, C}, "CNNBuilder 2D");
        mode_         = Mode::D2;
        current_shape_ = {C, H, W};
        pushHistory("Input");
    }

    // 3D explicite
    CNNBuilder(int D, int H, int W, int C) {
        requirePositive({D, H, W, C}, "CNNBuilder 3D");
        mode_          = Mode::D3;
        current_shape_ = {C, D, H, W};
        pushHistory("Input");
    }

    // Depuis un Tensor unifié — mode déduit de ndim()
    explicit CNNBuilder(const Tensor& input) {
        if (input.ndim() == 4) {
            mode_          = Mode::D2;
            current_shape_ = {input.dim(1), input.dim(2), input.dim(3)};
        } else {
            mode_          = Mode::D3;
            current_shape_ = {input.dim(1), input.dim(2), input.dim(3), input.dim(4)};
        }
        pushHistory("Input");
    }

    bool is3D() const { return mode_ == Mode::D3; }
    bool is2D() const { return mode_ == Mode::D2; }

    // =========================================================================
    // Convolution 2D
    // =========================================================================

    CNNBuilder& addConv(int out_channels,
                        int kernel_h, int kernel_w,
                        int stride_h, int stride_w,
                        int pad_h,    int pad_w,
                        int dilation_h = 1, int dilation_w = 1) {
        require2D("addConv");
        validateParams("addConv", {kernel_h, kernel_w}, {stride_h, stride_w});

        const int out_h = outputDim(H(), pad_h, dilatedK(kernel_h, dilation_h), stride_h);
        const int out_w = outputDim(W(), pad_w, dilatedK(kernel_w, dilation_w), stride_w);
        validateOutput("addConv", {out_h, out_w});

        std::cout << "Conv2D : " << C() << "→" << out_channels
                  << "  " << H() << "x" << W() << " → " << out_h << "x" << out_w
                  << "  (k=" << kernel_h << "x" << kernel_w
                  << " s=" << stride_h << "x" << stride_w
                  << " p=" << pad_h << "x" << pad_w;
        if (dilation_h != 1 || dilation_w != 1)
            std::cout << " d=" << dilation_h << "x" << dilation_w;
        std::cout << ")\n";

        current_shape_ = {out_channels, out_h, out_w};
        pushHistory("Conv2D");
        return *this;
    }

    // Kernel carré
    CNNBuilder& addConv(int out_channels, int kernel,
                        int stride = 1, int padding = 0, int dilation = 1) {
        return addConv(out_channels,
                       kernel, kernel, stride, stride,
                       padding, padding, dilation, dilation);
    }

    // Padding "same" — préserve H×W (stride=1)
    CNNBuilder& addConvSame(int out_channels, int kernel, int stride = 1) {
        const int pad = (kernel - 1) / 2;
        if ((H() + 2 * pad - kernel) % stride != 0)
            std::cerr << "[CNNBuilder] attention : 'same' peut ne pas préserver exactement la taille\n";
        return addConv(out_channels, kernel, stride, pad);
    }

    // =========================================================================
    // Convolution 3D
    // =========================================================================

    CNNBuilder& addConv3D(int out_channels,
                          int kernel_d, int kernel_h, int kernel_w,
                          int stride_d, int stride_h, int stride_w,
                          int pad_d,    int pad_h,    int pad_w,
                          int dilation_d = 1, int dilation_h = 1, int dilation_w = 1) {
        require3D("addConv3D");
        validateParams("addConv3D",
                       {kernel_d, kernel_h, kernel_w},
                       {stride_d, stride_h, stride_w});

        const int out_d = outputDim(D(), pad_d, dilatedK(kernel_d, dilation_d), stride_d);
        const int out_h = outputDim(H(), pad_h, dilatedK(kernel_h, dilation_h), stride_h);
        const int out_w = outputDim(W(), pad_w, dilatedK(kernel_w, dilation_w), stride_w);
        validateOutput("addConv3D", {out_d, out_h, out_w});

        std::cout << "Conv3D : " << C() << "→" << out_channels
                  << "  " << D() << "x" << H() << "x" << W()
                  << " → " << out_d << "x" << out_h << "x" << out_w
                  << "  (k=" << kernel_d << "x" << kernel_h << "x" << kernel_w
                  << " s=" << stride_d << "x" << stride_h << "x" << stride_w
                  << " p=" << pad_d << "x" << pad_h << "x" << pad_w;
        if (dilation_d != 1 || dilation_h != 1 || dilation_w != 1)
            std::cout << " d=" << dilation_d << "x" << dilation_h << "x" << dilation_w;
        std::cout << ")\n";

        current_shape_ = {out_channels, out_d, out_h, out_w};
        pushHistory("Conv3D");
        return *this;
    }

    // Kernel cubique
    CNNBuilder& addConv3D(int out_channels, int kernel,
                          int stride = 1, int padding = 0, int dilation = 1) {
        return addConv3D(out_channels,
                         kernel, kernel, kernel,
                         stride, stride, stride,
                         padding, padding, padding,
                         dilation, dilation, dilation);
    }

    // =========================================================================
    // Pooling 2D
    // =========================================================================

    CNNBuilder& addMaxPool(int pool_h, int pool_w,
                           int stride_h, int stride_w,
                           int pad_h = 0, int pad_w = 0) {
        require2D("addMaxPool");
        validateParams("addMaxPool", {pool_h, pool_w}, {stride_h, stride_w});

        const int out_h = outputDim(H(), pad_h, pool_h, stride_h);
        const int out_w = outputDim(W(), pad_w, pool_w, stride_w);
        validateOutput("addMaxPool", {out_h, out_w});

        std::cout << "MaxPool2D : " << H() << "x" << W()
                  << " → " << out_h << "x" << out_w
                  << "  (pool=" << pool_h << "x" << pool_w
                  << " s=" << stride_h << "x" << stride_w << ")\n";

        current_shape_ = {C(), out_h, out_w};
        pushHistory("MaxPool2D");
        return *this;
    }

    // Kernel carré, stride = pool par défaut
    CNNBuilder& addMaxPool(int pool, int stride = -1, int padding = 0) {
        if (stride == -1) stride = pool;
        return addMaxPool(pool, pool, stride, stride, padding, padding);
    }

    CNNBuilder& addAvgPool(int pool, int stride = -1, int padding = 0) {
        require2D("addAvgPool");
        if (stride == -1) stride = pool;
        validateParams("addAvgPool", {pool, pool}, {stride, stride});

        const int out_h = outputDim(H(), padding, pool, stride);
        const int out_w = outputDim(W(), padding, pool, stride);
        validateOutput("addAvgPool", {out_h, out_w});

        std::cout << "AvgPool2D : " << H() << "x" << W()
                  << " → " << out_h << "x" << out_w << "\n";

        current_shape_ = {C(), out_h, out_w};
        pushHistory("AvgPool2D");
        return *this;
    }

    CNNBuilder& addGlobalAvgPool() {
        if (is3D()) {
            std::cout << "GlobalAvgPool3D : " << C() << "x" << D() << "x" << H() << "x" << W()
                      << " → " << C() << "x1x1x1\n";
            current_shape_ = {C(), 1, 1, 1};
            pushHistory("GlobalAvgPool3D");
        } else {
            std::cout << "GlobalAvgPool2D : " << C() << "x" << H() << "x" << W()
                      << " → " << C() << "x1x1\n";
            current_shape_ = {C(), 1, 1};
            pushHistory("GlobalAvgPool2D");
        }
        return *this;
    }

    // =========================================================================
    // Pooling 3D
    // =========================================================================

    CNNBuilder& addMaxPool3D(int pool_d, int pool_h, int pool_w,
                             int stride_d, int stride_h, int stride_w,
                             int pad_d = 0, int pad_h = 0, int pad_w = 0) {
        require3D("addMaxPool3D");
        validateParams("addMaxPool3D",
                       {pool_d, pool_h, pool_w},
                       {stride_d, stride_h, stride_w});

        const int out_d = outputDim(D(), pad_d, pool_d, stride_d);
        const int out_h = outputDim(H(), pad_h, pool_h, stride_h);
        const int out_w = outputDim(W(), pad_w, pool_w, stride_w);
        validateOutput("addMaxPool3D", {out_d, out_h, out_w});

        std::cout << "MaxPool3D : " << D() << "x" << H() << "x" << W()
                  << " → " << out_d << "x" << out_h << "x" << out_w
                  << "  (pool=" << pool_d << "x" << pool_h << "x" << pool_w
                  << " s=" << stride_d << "x" << stride_h << "x" << stride_w << ")\n";

        current_shape_ = {C(), out_d, out_h, out_w};
        pushHistory("MaxPool3D");
        return *this;
    }

    // Kernel cubique, stride = pool par défaut
    CNNBuilder& addMaxPool3D(int pool, int stride = -1, int padding = 0) {
        if (stride == -1) stride = pool;
        return addMaxPool3D(pool, pool, pool, stride, stride, stride,
                            padding, padding, padding);
    }

    // =========================================================================
    // Couches denses (communes 2D/3D)
    // =========================================================================

    CNNBuilder& addFlatten() {
        const int flat = flattenSize();
        if (is3D()) {
            std::cout << "Flatten : " << C() << "x" << D() << "x" << H() << "x" << W()
                      << " → " << flat << "\n";
        } else {
            std::cout << "Flatten : " << C() << "x" << H() << "x" << W()
                      << " → " << flat << "\n";
        }
        first_dense_input_ = flat;
        // Après flatten, on repasse en mode 2D avec shape {flat, 1, 1}
        mode_          = Mode::D2;
        current_shape_ = {flat, 1, 1};
        pushHistory("Flatten");
        return *this;
    }

    CNNBuilder& addDense(int output_size) {
        if (output_size <= 0)
            throw std::invalid_argument("addDense: output_size doit être positif");
        std::cout << "Dense : " << C() << " → " << output_size << "\n";
        if (first_dense_input_ == 0) first_dense_input_ = C();
        current_shape_ = {output_size, 1, 1};
        pushHistory("Dense");
        return *this;
    }

    // =========================================================================
    // Accesseurs
    // =========================================================================

    int flattenSize() const {
        int s = 1;
        for (int v : current_shape_) s *= v;
        return s;
    }

    int firstDenseInputSize() const { return first_dense_input_; }

    std::vector<int> currentShape() const { return current_shape_; }

    // =========================================================================
    // Affichage
    // =========================================================================

    void printCurrentShape() const {
        std::cout << "▶ Forme courante : ";
        if (is3D())
            std::cout << C() << " × " << D() << " × " << H() << " × " << W()
                      << "  (C×D×H×W)\n";
        else
            std::cout << C() << " × " << H() << " × " << W()
                      << "  (C×H×W)\n";
    }

    void printArchitecture() const {
        const std::string bar(55, '=');
        std::cout << "\n" << bar << "\n"
                  << "          ARCHITECTURE CNN "
                  << (is3D() ? "3D" : "2D") << "\n"
                  << bar << "\n";

        for (const auto& entry : history_) {
            std::cout << std::left << std::setw(18) << entry.name << ": ";
            const auto& s = entry.shape;
            if (s.size() == 3 && s[1] == 1 && s[2] == 1) {
                std::cout << s[0];                              // Dense/Flatten
            } else if (s.size() == 3) {
                std::cout << s[0] << "x" << s[1] << "x" << s[2]; // C×H×W
            } else if (s.size() == 4) {
                std::cout << s[0] << "x" << s[1] << "x"          // C×D×H×W
                          << s[2] << "x" << s[3];
            }
            std::cout << "\n";
        }
        std::cout << bar << "\n";
    }

private:
    enum class Mode { D2, D3 };
    Mode             mode_              = Mode::D2;
    std::vector<int> current_shape_;                 // {C,H,W} ou {C,D,H,W}
    int              first_dense_input_ = 0;

    struct HistoryEntry { std::string name; std::vector<int> shape; };
    std::vector<HistoryEntry> history_;

    // ── Accesseurs internes ────────────────────────────────────────────────
    int C() const { return current_shape_[0]; }
    // En mode 3D : {C, D, H, W}
    // En mode 2D : {C, H, W}
    int D() const { return (mode_ == Mode::D3) ? current_shape_[1] : 1; }
    int H() const { return (mode_ == Mode::D3) ? current_shape_[2] : current_shape_[1]; }
    int W() const { return (mode_ == Mode::D3) ? current_shape_[3] : current_shape_[2]; }

    // ── Calculs ───────────────────────────────────────────────────────────
    static int outputDim(int in, int pad, int kernel, int stride) {
        return (in + 2 * pad - kernel) / stride + 1;
    }
    static int dilatedK(int k, int d) { return k + (k - 1) * (d - 1); }

    // ── Validation ────────────────────────────────────────────────────────
    static void requirePositive(std::initializer_list<int> vals, const std::string& op) {
        for (int v : vals)
            if (v <= 0) throw std::invalid_argument(op + ": toutes les dimensions doivent être positives");
    }

    void require2D(const std::string& op) const {
        if (mode_ == Mode::D3)
            throw std::logic_error(op + ": opération 2D appelée sur un builder 3D — utilisez la variante *3D");
    }

    void require3D(const std::string& op) const {
        if (mode_ == Mode::D2)
            throw std::logic_error(op + ": opération 3D appelée sur un builder 2D — utilisez la variante sans *3D");
    }

    static void validateParams(const std::string& op,
                               std::initializer_list<int> kernels,
                               std::initializer_list<int> strides) {
        for (int k : kernels)
            if (k <= 0) throw std::invalid_argument(op + ": kernel doit être positif");
        for (int s : strides)
            if (s <= 0) throw std::invalid_argument(op + ": stride doit être positif");
    }

    static void validateOutput(const std::string& op, std::initializer_list<int> dims) {
        for (int d : dims)
            if (d <= 0)
                throw std::runtime_error(op + ": dimension de sortie invalide (" + std::to_string(d) + ")");
    }

    // ── Historique ────────────────────────────────────────────────────────
    void pushHistory(const std::string& name) {
        history_.push_back({name, current_shape_});
    }
};