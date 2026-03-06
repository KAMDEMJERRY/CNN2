#pragma once
#include "Layer.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// ReLULayer
// Fonctionne en mode 4D et 5D grâce à l'accès plat operator[]
// Le masque utilise l'accès plat — pas de dépendance au rang
// ─────────────────────────────────────────────────────────────────────────────
class ReLULayer : public Layer {
private:
    Tensor input_cache;

public:
    ReLULayer()  = default;
    ~ReLULayer() override = default;

    Tensor forward(const Tensor& input) override {
        input_cache = input;
        Tensor output = input;

        // Accès plat : indépendant du rang (4D ou 5D)
        for (int i = 0; i < output.size(); ++i)
            output[i] = (output[i] > 0.0f) ? output[i] : 0.0f;

        return output;
    }

    Tensor backward(const Tensor& gradOutput) override {
        Tensor gradInput(gradOutput.shape());

        // Masque binaire via accès plat : 1 si input > 0, 0 sinon
        for (int i = 0; i < gradOutput.size(); ++i)
            gradInput[i] = (input_cache[i] > 0.0f) ? gradOutput[i] : 0.0f;

        return gradInput;
    }

    std::string getName() const override { return "ReLU"; }
};

// ─────────────────────────────────────────────────────────────────────────────
// LeakyReLULayer
// Inchangé dans la logique — accès plat déjà utilisé, compatible 4D/5D
// ─────────────────────────────────────────────────────────────────────────────
class LeakyReLULayer : public Layer {
private:
    Tensor input_cache;
    float  alpha;

public:
    explicit LeakyReLULayer(float alpha = 0.01f) : alpha(alpha) {}
    ~LeakyReLULayer() override = default;

    Tensor forward(const Tensor& input) override {
        input_cache = input;
        Tensor output = input;

        for (int i = 0; i < input.size(); ++i)
            output[i] = (input[i] > 0.0f) ? input[i] : alpha * input[i];

        return output;
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (gradOutput.shape() != input_cache.shape())
            throw std::runtime_error("[LeakyReLU] shape mismatch in backward");

        Tensor gradInput(gradOutput.shape());

        for (int i = 0; i < gradOutput.size(); ++i)
            gradInput[i] = (input_cache[i] > 0.0f) ? gradOutput[i]
                                                    : alpha * gradOutput[i];

        return gradInput;
    }

    std::string getName() const override { return "LeakyReLU"; }
};

// ─────────────────────────────────────────────────────────────────────────────
// SigmoidLayer
// Correction : le backward utilisait gradInput[i] au lieu d'input_cache[i]
// pour recalculer sigmoid — résultat incorrect dans votre version originale
// ─────────────────────────────────────────────────────────────────────────────
class SigmoidLayer : public Layer {
private:
    Tensor output_cache;   // cache de la sortie forward pour le backward

public:
    SigmoidLayer()  = default;
    ~SigmoidLayer() override = default;

    Tensor forward(const Tensor& input) override {
        Tensor output = input;

        for (int i = 0; i < output.size(); ++i)
            output[i] = 1.0f / (1.0f + std::exp(-input[i]));

        output_cache = output;   // stocker σ(x) pour le backward
        return output;
    }

    Tensor backward(const Tensor& gradOutput) override {
        // dL/dx = dL/dy * σ(x) * (1 - σ(x))
        // σ(x) est dans output_cache — pas besoin de recalculer
        Tensor gradInput(gradOutput.shape());

        for (int i = 0; i < gradOutput.size(); ++i) {
            float s = output_cache[i];
            gradInput[i] = gradOutput[i] * s * (1.0f - s);
        }

        return gradInput;
    }

    std::string getName() const override { return "Sigmoid"; }
};

// ─────────────────────────────────────────────────────────────────────────────
// SoftmaxLayer
//
// Supporte 4D et 5D :
//   Mode 4D : entrée (B, C, 1, 1)       — classification standard
//   Mode 5D : entrée (B, C, 1, 1, 1)    — classification après GlobalAvgPool3D
//
// Le softmax est toujours appliqué sur la dimension des canaux (dim 1).
// Les dimensions spatiales (H, W [, D]) doivent toutes valoir 1.
// ─────────────────────────────────────────────────────────────────────────────
class SoftmaxLayer : public Layer {
private:
    Tensor output_cache;

public:
    SoftmaxLayer()  = default;
    ~SoftmaxLayer() override = default;

    Tensor forward(const Tensor& input) override {
        validateSpatialDims(input, "forward");

        int B = input.dim(0);
        int C = input.dim(1);

        Tensor output(input.shape());

#pragma omp parallel for
        for (int b = 0; b < B; ++b) {
            // Stabilité numérique : soustraire le max avant exp
            float max_val = -std::numeric_limits<float>::infinity();
            for (int c = 0; c < C; ++c)
                max_val = std::max(max_val, flatAccess(input, b, c));

            float sum_exp = 0.0f;
            for (int c = 0; c < C; ++c) {
                float e = std::exp(flatAccess(input, b, c) - max_val);
                flatAccess(output, b, c) = e;
                sum_exp += e;
            }

            for (int c = 0; c < C; ++c)
                flatAccess(output, b, c) /= sum_exp;
        }

        output_cache = output;
        return output;
    }

    Tensor backward(const Tensor& gradOutput) override {
        validateSpatialDims(gradOutput, "backward");

        int B = gradOutput.dim(0);
        int C = gradOutput.dim(1);

        Tensor gradInput(gradOutput.shape());

#pragma omp parallel for
        for (int b = 0; b < B; ++b) {
            // grad_input = s * (grad_output - dot(s, grad_output))
            float dot = 0.0f;
            for (int c = 0; c < C; ++c)
                dot += flatAccess(output_cache, b, c) * flatAccess(gradOutput, b, c);

            for (int c = 0; c < C; ++c) {
                float s = flatAccess(output_cache, b, c);
                flatAccess(gradInput, b, c) = s * (flatAccess(gradOutput, b, c) - dot);
            }
        }

        return gradInput;
    }

    std::string getName() const override { return "Softmax"; }

    // Argmax des probabilités → classe prédite par échantillon
    Tensor getPredictions() const {
        int B = output_cache.dim(0);
        int C = output_cache.dim(1);

        // Résultat : (B, 1, 1, 1) en mode 4D ou (B, 1, 1, 1, 1) en mode 5D
        Tensor predictions(output_cache.shape());
        predictions.setZero();

        for (int b = 0; b < B; ++b) {
            int   max_class = 0;
            float max_prob  = flatAccess(output_cache, b, 0);

            for (int c = 1; c < C; ++c) {
                float p = flatAccess(output_cache, b, c);
                if (p > max_prob) { max_prob = p; max_class = c; }
            }
            flatAccess(predictions, b, 0) = static_cast<float>(max_class);
        }

        return predictions;
    }

    const Tensor& getProbabilities() const { return output_cache; }

private:

    // ── Accès unifié sur la dimension canal, indépendant du rang ─────────────
    //
    // En 4D : t(b, c, 0, 0)
    // En 5D : t(b, c, 0, 0, 0)
    //
    // Utilise l'index plat pour éviter toute dépendance au rang
    // Position dans le stockage RowMajor : b*(C*1*1[*1]) + c*(1*1[*1])

    static float& flatAccess(Tensor& t, int b, int c) {
        // C × spatial_vol éléments par batch
        // spatial_vol = 1 pour (B,C,1,1) ou (B,C,1,1,1)
        int C     = t.dim(1);
        int index = b * C + c;   // spatial = 1 → stride=1
        return t[index];
    }

    static float flatAccess(const Tensor& t, int b, int c) {
        int C     = t.dim(1);
        int index = b * C + c;
        return t[index];
    }

    // Vérifie que toutes les dimensions spatiales valent 1
    static void validateSpatialDims(const Tensor& t, const std::string& where) {
        bool ok = true;
        if (t.ndim() == 4) {
            ok = (t.dim(2) == 1 && t.dim(3) == 1);
        } else {
            ok = (t.dim(2) == 1 && t.dim(3) == 1 && t.dim(4) == 1);
        }
        if (!ok)
            throw std::runtime_error(
                "[SoftmaxLayer::" + where + "] "
                "Les dimensions spatiales doivent toutes valoir 1. "
                "Ajoutez un GlobalAvgPool avant Softmax.");
    }
};