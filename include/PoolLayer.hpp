#pragma once
#include "Layer.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// MaxPoolLayer  (2D — mode 4D)
// Refactorisé depuis votre version originale :
//   - pool_size_h / pool_size_w séparés (était pool_size carré uniquement)
//   - stride_h / stride_w séparés
//   - compatible Tensor unifié (mode 4D)
// ─────────────────────────────────────────────────────────────────────────────
class MaxPoolLayer : public Layer {
private:
    int pool_h, pool_w;
    int stride_h, stride_w;

    Tensor input_cache;

    // max_indices[flat_idx] = {ih, iw} — position du max dans l'input
    std::vector<std::array<int, 2>> max_indices;

    struct OutDims2D { int h, w; };

    OutDims2D computeOutputDims(int in_h, int in_w) const {
        return {
            static_cast<int>(std::floor((in_h - pool_h) / static_cast<float>(stride_h))) + 1,
            static_cast<int>(std::floor((in_w - pool_w) / static_cast<float>(stride_w))) + 1
        };
    }

public:
    // pool_size carré, stride unique — conserve votre interface d'origine
    explicit MaxPoolLayer(int pool_size = 2, int stride = 2)
        : pool_h(pool_size), pool_w(pool_size)
        , stride_h(stride),  stride_w(stride) {}

    // Version rectangulaire
    MaxPoolLayer(int pool_h, int pool_w, int stride_h, int stride_w)
        : pool_h(pool_h), pool_w(pool_w)
        , stride_h(stride_h), stride_w(stride_w) {}

    ~MaxPoolLayer() override = default;

    Tensor forward(const Tensor& input) override {
        if (input.ndim() != 4)
            throw std::runtime_error("[MaxPoolLayer] Attend un Tensor 4D (B,C,H,W)");

        input_cache = input;

        int B = input.dim(0);
        int C = input.dim(1);
        int H = input.dim(2);
        int W = input.dim(3);

        auto od = computeOutputDims(H, W);

        Tensor output(B, C, od.h, od.w);

        max_indices.clear();
        max_indices.resize(B * C * od.h * od.w);

        for (int b  = 0; b  < B;    ++b)
        for (int c  = 0; c  < C;    ++c)
        for (int oh = 0; oh < od.h; ++oh)
        for (int ow = 0; ow < od.w; ++ow) {
            int sh = oh * stride_h;
            int sw = ow * stride_w;

            float max_val = std::numeric_limits<float>::lowest();
            int   max_ih  = sh, max_iw = sw;

            for (int ph = 0; ph < pool_h; ++ph)
            for (int pw = 0; pw < pool_w; ++pw) {
                int ih = sh + ph;
                int iw = sw + pw;
                if (ih < H && iw < W) {
                    float v = input(b, c, ih, iw);
                    if (v > max_val) { max_val = v; max_ih = ih; max_iw = iw; }
                }
            }

            output(b, c, oh, ow) = max_val;

            int idx = ((b * C + c) * od.h + oh) * od.w + ow;
            max_indices[idx] = {max_ih, max_iw};
        }

        return output;
    }

    Tensor backward(const Tensor& grad_output) override {
        int B = input_cache.dim(0);
        int C = input_cache.dim(1);
        int H = input_cache.dim(2);
        int W = input_cache.dim(3);

        auto od = computeOutputDims(H, W);

        Tensor grad_input(B, C, H, W);
        grad_input.setZero();

        for (int b  = 0; b  < B;    ++b)
        for (int c  = 0; c  < C;    ++c)
        for (int oh = 0; oh < od.h; ++oh)
        for (int ow = 0; ow < od.w; ++ow) {
            int idx  = ((b * C + c) * od.h + oh) * od.w + ow;
            int max_ih = max_indices[idx][0];
            int max_iw = max_indices[idx][1];

            grad_input(b, c, max_ih, max_iw) += grad_output(b, c, oh, ow);
        }

        return grad_input;
    }

    std::string getName() const override { return "MaxPool2D"; }
};

// ─────────────────────────────────────────────────────────────────────────────
// MaxPool3DLayer  (mode 5D)
// Extension directe de MaxPoolLayer à la dimension profondeur
// ─────────────────────────────────────────────────────────────────────────────
class MaxPool3DLayer : public Layer {
private:
    int pool_d, pool_h, pool_w;
    int stride_d, stride_h, stride_w;

    Tensor input_cache;

    // max_indices[flat_idx] = {id, ih, iw}
    std::vector<std::array<int, 3>> max_indices;

    struct OutDims3D { int d, h, w; };

    OutDims3D computeOutputDims(int in_d, int in_h, int in_w) const {
        return {
            static_cast<int>(std::floor((in_d - pool_d) / static_cast<float>(stride_d))) + 1,
            static_cast<int>(std::floor((in_h - pool_h) / static_cast<float>(stride_h))) + 1,
            static_cast<int>(std::floor((in_w - pool_w) / static_cast<float>(stride_w))) + 1
        };
    }

public:
    // Cube isotrope — cas le plus courant en médical (volumes isotropes)
    explicit MaxPool3DLayer(int pool_size = 2, int stride = 2)
        : pool_d(pool_size), pool_h(pool_size), pool_w(pool_size)
        , stride_d(stride),  stride_h(stride),  stride_w(stride) {}

    // Version anisotrope — utile pour les CT (résolution z différente)
    MaxPool3DLayer(int pool_d,   int pool_h,   int pool_w,
                   int stride_d, int stride_h, int stride_w)
        : pool_d(pool_d), pool_h(pool_h), pool_w(pool_w)
        , stride_d(stride_d), stride_h(stride_h), stride_w(stride_w) {}

    ~MaxPool3DLayer() override = default;

    Tensor forward(const Tensor& input) override {
        if (input.ndim() != 5)
            throw std::runtime_error("[MaxPool3DLayer] Attend un Tensor 5D (B,C,D,H,W)");

        input_cache = input;

        int B = input.dim(0);
        int C = input.dim(1);
        int D = input.dim(2);
        int H = input.dim(3);
        int W = input.dim(4);

        auto od = computeOutputDims(D, H, W);

        Tensor output(B, C, od.d, od.h, od.w);

        max_indices.clear();
        max_indices.resize(B * C * od.d * od.h * od.w);

        for (int b   = 0; b   < B;    ++b)
        for (int c   = 0; c   < C;    ++c)
        for (int od_ = 0; od_ < od.d; ++od_)
        for (int oh  = 0; oh  < od.h; ++oh)
        for (int ow  = 0; ow  < od.w; ++ow) {
            int sd = od_ * stride_d;
            int sh = oh  * stride_h;
            int sw = ow  * stride_w;

            float max_val = std::numeric_limits<float>::lowest();
            int   max_id  = sd, max_ih = sh, max_iw = sw;

            for (int pd = 0; pd < pool_d; ++pd)
            for (int ph = 0; ph < pool_h; ++ph)
            for (int pw = 0; pw < pool_w; ++pw) {
                int id = sd + pd;
                int ih = sh + ph;
                int iw = sw + pw;
                if (id < D && ih < H && iw < W) {
                    float v = input(b, c, id, ih, iw);
                    if (v > max_val) {
                        max_val = v;
                        max_id = id; max_ih = ih; max_iw = iw;
                    }
                }
            }

            output(b, c, od_, oh, ow) = max_val;

            int idx = ((b * C + c) * od.d + od_) * od.h * od.w
                    + oh * od.w + ow;
            max_indices[idx] = {max_id, max_ih, max_iw};
        }

        return output;
    }

    Tensor backward(const Tensor& grad_output) override {
        int B = input_cache.dim(0);
        int C = input_cache.dim(1);
        int D = input_cache.dim(2);
        int H = input_cache.dim(3);
        int W = input_cache.dim(4);

        auto od = computeOutputDims(D, H, W);

        Tensor grad_input(B, C, D, H, W);
        grad_input.setZero();

        for (int b   = 0; b   < B;    ++b)
        for (int c   = 0; c   < C;    ++c)
        for (int od_ = 0; od_ < od.d; ++od_)
        for (int oh  = 0; oh  < od.h; ++oh)
        for (int ow  = 0; ow  < od.w; ++ow) {
            int idx = ((b * C + c) * od.d + od_) * od.h * od.w
                    + oh * od.w + ow;

            int max_id = max_indices[idx][0];
            int max_ih = max_indices[idx][1];
            int max_iw = max_indices[idx][2];

            grad_input(b, c, max_id, max_ih, max_iw) +=
                grad_output(b, c, od_, oh, ow);
        }

        return grad_input;
    }

    std::string getName() const override { return "MaxPool3D"; }
};

// ─────────────────────────────────────────────────────────────────────────────
// GlobalAvgPool2DLayer  (mode 4D)
// Réduit (B, C, H, W) → (B, C, 1, 1)
// Pas de paramètres, pas de backward complexe
// ─────────────────────────────────────────────────────────────────────────────
class GlobalAvgPool2DLayer : public Layer {
private:
    int cached_H = 0;
    int cached_W = 0;

public:
    GlobalAvgPool2DLayer()  = default;
    ~GlobalAvgPool2DLayer() override = default;

    Tensor forward(const Tensor& input) override {
        if (input.ndim() != 4)
            throw std::runtime_error("[GlobalAvgPool2D] Attend un Tensor 4D (B,C,H,W)");

        int B = input.dim(0);
        int C = input.dim(1);
        int H = input.dim(2);
        int W = input.dim(3);

        cached_H = H;
        cached_W = W;

        Tensor output(B, C, 1, 1);
        output.setZero();

        for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c) {
            float sum = 0.f;
            for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                sum += input(b, c, h, w);
            output(b, c, 0, 0) = sum / (H * W);
        }

        return output;
    }

    // Backward : redistribue uniformément le gradient sur H×W
    Tensor backward(const Tensor& grad_output) override {
        int B = grad_output.dim(0);
        int C = grad_output.dim(1);

        Tensor grad_input(B, C, cached_H, cached_W);
        float scale = 1.0f / (cached_H * cached_W);

        for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c) {
            float g = grad_output(b, c, 0, 0) * scale;
            for (int h = 0; h < cached_H; ++h)
            for (int w = 0; w < cached_W; ++w)
                grad_input(b, c, h, w) = g;
        }

        return grad_input;
    }

    std::string getName() const override { return "GlobalAvgPool2D"; }
};

// ─────────────────────────────────────────────────────────────────────────────
// GlobalAvgPool3DLayer  (mode 5D)
// Réduit (B, C, D, H, W) → (B, C, 1, 1, 1)
// Brique essentielle avant SoftmaxLayer dans votre architecture
// ─────────────────────────────────────────────────────────────────────────────
class GlobalAvgPool3DLayer : public Layer {
private:
    int cached_D = 0;
    int cached_H = 0;
    int cached_W = 0;

public:
    GlobalAvgPool3DLayer()  = default;
    ~GlobalAvgPool3DLayer() override = default;

    Tensor forward(const Tensor& input) override {
        if (input.ndim() != 5)
            throw std::runtime_error("[GlobalAvgPool3D] Attend un Tensor 5D (B,C,D,H,W)");

        int B = input.dim(0);
        int C = input.dim(1);
        int D = input.dim(2);
        int H = input.dim(3);
        int W = input.dim(4);

        cached_D = D;
        cached_H = H;
        cached_W = W;

        // Sortie (B, C, 1, 1, 1) — prête pour SoftmaxLayer
        Tensor output(B, C, 1, 1, 1);
        output.setZero();

        for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c) {
            float sum = 0.f;
            for (int d = 0; d < D; ++d)
            for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                sum += input(b, c, d, h, w);
            output(b, c, 0, 0, 0) = sum / (D * H * W);
        }

        return output;
    }

    // Backward : redistribue uniformément le gradient sur D×H×W
    Tensor backward(const Tensor& grad_output) override {
        int B = grad_output.dim(0);
        int C = grad_output.dim(1);

        Tensor grad_input(B, C, cached_D, cached_H, cached_W);
        float scale = 1.0f / (cached_D * cached_H * cached_W);

        for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c) {
            float g = grad_output(b, c, 0, 0, 0) * scale;
            for (int d = 0; d < cached_D; ++d)
            for (int h = 0; h < cached_H; ++h)
            for (int w = 0; w < cached_W; ++w)
                grad_input(b, c, d, h, w) = g;
        }

        return grad_input;
    }

    std::string getName() const override { return "GlobalAvgPool3D"; }
};