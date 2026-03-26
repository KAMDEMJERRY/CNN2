#pragma once
#include <string>
#include <vector>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include "Tensor.hpp"
#include "NpyParser.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// Split disponibles dans un dataset MedMNIST3D
// ─────────────────────────────────────────────────────────────────────────────
enum class Split { TRAIN, VAL, TEST };

// ─────────────────────────────────────────────────────────────────────────────
// MedMNIST3DDataset
// Équivalent 3D de ImageFolderDataset — charge les .npy en mémoire
// ─────────────────────────────────────────────────────────────────────────────
class MedMNIST3DDataset {
private:

    // Volumes chargés en RAM : (N, 1, D, H, W) float32
    Tensor images;

    // Labels scalaires
    std::vector<int> labels;

    // Métadonnées
    int num_classes;
    std::string dataset_name;
    Split split;

    // Statistiques de normalisation
    bool   normalized    = false;
    float  vol_mean      = 0.0f;
    float  vol_std       = 1.0f;

public:

    // ── Constructeur ──────────────────────────────────────────────────────────

    // root_dir  : dossier contenant .npy, train_labels.npy, ...
    // split     : TRAIN, VAL ou TEST
    // num_classes : 2 pour Adrenal, 3 pour Fracture
    // normalize : applique z-score sur les volumes après chargement
    MedMNIST3DDataset(const std::string& root_dir,
                      Split              split,
                      int                num_classes,
                      const std::string& name     = "MedMNIST3D",
                      bool               normalize = true,
                      bool               verbose   = true)
        : num_classes(num_classes), dataset_name(name), split(split)
    {
        std::string prefix = root_dir + "/" + splitPrefix(split);

        if (verbose) {
            std::cout << "═══════════════════════════════════════════════════════" << std::endl;
            std::cout << "📦 " << name << " — " << splitName(split) << std::endl;
            std::cout << "═══════════════════════════════════════════════════════" << std::endl;
        }

        // 1. Charger les images
        std::string images_path = prefix + "_images.npy";
        if (verbose) std::cout << "⏳ Chargement images : " << images_path << std::endl;
        images = NpyParser::load(images_path, verbose);
        
        if (verbose) {
            std::cout << "✅ Images chargées → ";
            images.printShape();
        }

        // 2. Charger les labels
        std::string labels_path = prefix + "_labels.npy";
        if (verbose) std::cout << "⏳ Chargement labels : " << labels_path << std::endl;
        labels = NpyParser::loadLabels(labels_path, verbose);

        if (verbose)
            std::cout << "✅ Labels chargés → " << labels.size() << " échantillons" << std::endl;

        // 3. Validation
        if (static_cast<int>(labels.size()) != images.dim(0))
            throw std::runtime_error("[MedMNIST3DDataset] Mismatch images/labels: "
                + std::to_string(images.dim(0)) + " vs "
                + std::to_string(labels.size()));

        // 4. Normalisation z-score (optionnelle)
        if (normalize) {
            computeAndNormalize(verbose);
        }

        if (verbose) printSummary();
    }

    // ── Accès aux données ─────────────────────────────────────────────────────

    int getNumSamples()  const { return images.dim(0); }
    int getNumClasses()  const { return num_classes; }
    int getDepth()       const { return images.dim(2); }
    int getHeight()      const { return images.dim(3); }
    int getWidth()       const { return images.dim(4); }
    int getNumChannels() const { return images.dim(1); }

    int getLabel(int idx) const {
        if (idx < 0 || idx >= static_cast<int>(labels.size()))
            throw std::out_of_range("[MedMNIST3DDataset] Index hors limites");
        return labels[idx];
    }

    const std::vector<int>& getLabels() const { return labels; }

    // Retourne un volume individuel : Tensor (1, C, D, H, W)
    Tensor getVolume(int idx) const {
        if (idx < 0 || idx >= getNumSamples())
            throw std::out_of_range("[MedMNIST3DDataset] Index hors limites");

        int C = images.dim(1);
        int D = images.dim(2);
        int H = images.dim(3);
        int W = images.dim(4);

        Tensor vol(1, C, D, H, W);
        int vol_size = C * D * H * W;
        const float* src = images.getData() + idx * vol_size;
        std::memcpy(vol.getData(), src, vol_size * sizeof(float));
        return vol;
    }

    // ── getBatch ──────────────────────────────────────────────────────────────

    // Retourne (images_batch, labels_onehot)
    // images_batch : Tensor (B, C, D, H, W)
    // labels_onehot: Tensor (B, num_classes, 1, 1, 1)
    std::pair< Tensor, Tensor> getBatch(const std::vector<int>& indices) const {
        int B = static_cast<int>(indices.size());
        int C = images.dim(1);
        int D = images.dim(2);
        int H = images.dim(3);
        int W = images.dim(4);

        Tensor batch_images(B, C, D, H, W);
        int vol_size = C * D * H * W;

        for (int i = 0; i < B; ++i) {
            int idx = indices[i];
            if (idx < 0 || idx >= getNumSamples())
                throw std::out_of_range("[MedMNIST3DDataset] Index hors limites dans getBatch");

            const float* src = images.getData() + idx * vol_size;
            float*       dst = batch_images.getData() + i * vol_size;
            std::memcpy(dst, src, vol_size * sizeof(float));
        }

        // Labels → one-hot
        std::vector<int> batch_labels;
        batch_labels.reserve(B);
        for (int idx : indices) batch_labels.push_back(labels[idx]);

        Tensor batch_labels_onehot = labelsToOneHot(batch_labels);

        return { batch_images, batch_labels_onehot };
    }

    // ── One-hot ───────────────────────────────────────────────────────────────

    // Retourne Tensor (N, num_classes, 1, 1, 1)
    Tensor labelsToOneHot(const std::vector<int>& lbls) const {
        int N = static_cast<int>(lbls.size());
        Tensor onehot(N, num_classes, 1, 1, 1);
        onehot.setZero();
        for (int i = 0; i < N; ++i) {
            if (lbls[i] >= 0 && lbls[i] < num_classes)
                onehot(i, lbls[i], 0, 0, 0) = 1.0f;
        }
        return onehot;
    }

    // ── Accuracy ──────────────────────────────────────────────────────────────

    float computeAccuracy(const Tensor& predictions, const Tensor& targets) const {
        int B = predictions.dim(0);
        int correct = 0;

        for (int b = 0; b < B; ++b) {
            // Argmax des prédictions
            int pred_class = 0;
            float max_val  = predictions(b, 0, 0, 0, 0);
            for (int c = 1; c < num_classes; ++c) {
                if (predictions(b, c, 0, 0, 0) > max_val) {
                    max_val    = predictions(b, c, 0, 0, 0);
                    pred_class = c;
                }
            }

            // Argmax des targets (one-hot)
            for (int c = 0; c < num_classes; ++c) {
                if (targets(b, c, 0, 0, 0) > 0.5f) {
                    if (c == pred_class) correct++;
                    break;
                }
            }
        }

        return static_cast<float>(correct) / B;
    }

    // ── Distribution des classes ──────────────────────────────────────────────

    void printClassDistribution() const {
        std::vector<int> counts(num_classes, 0);
        for (int l : labels) counts[l]++;

        std::cout << "📊 Distribution des classes (" << dataset_name << " — " << splitName(split) << "):" << std::endl;
        for (int c = 0; c < num_classes; ++c) {
            float pct = 100.0f * counts[c] / static_cast<int>(labels.size());
            std::cout << "   Classe " << c << " : "
                      << std::setw(5) << counts[c] << " samples ("
                      << std::fixed << std::setprecision(1) << pct << "%)" << std::endl;
        }
    }

    // ── Stats de normalisation ────────────────────────────────────────────────

    float getMean() const { return vol_mean; }
    float getStd()  const { return vol_std;  }

    // ── Résumé ────────────────────────────────────────────────────────────────

    void printSummary() const {
        std::cout << "───────────────────────────────────────────────────────" << std::endl;
        std::cout << "📋 " << dataset_name << " (" << splitName(split) << ")" << std::endl;
        std::cout << "   Samples     : " << getNumSamples()  << std::endl;
        std::cout << "   Classes     : " << num_classes       << std::endl;
        std::cout << "   Volume shape: ("
                  << getNumChannels() << ", "
                  << getDepth()   << ", "
                  << getHeight()  << ", "
                  << getWidth()   << ")" << std::endl;
        if (normalized) {
            std::cout << "   Mean (norm) : " << std::fixed << std::setprecision(6) << vol_mean << std::endl;
            std::cout << "   Std  (norm) : " << vol_std << std::endl;
        }
        std::cout << "───────────────────────────────────────────────────────" << std::endl;
        printClassDistribution();
        std::cout << "═══════════════════════════════════════════════════════" << std::endl;
    }

private:

    // ── Normalisation z-score ─────────────────────────────────────────────────

    void computeAndNormalize(bool verbose) {
        float* data = images.getData();
        size_t n    = images.size();

        // Calcul de la moyenne
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) sum += data[i];
        vol_mean = static_cast<float>(sum / n);

        // Calcul de l'écart-type
        double sum_sq = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double diff = data[i] - vol_mean;
            sum_sq += diff * diff;
        }
        vol_std = static_cast<float>(std::sqrt(sum_sq / n));
        if (vol_std < 1e-8f) vol_std = 1.0f; // sécurité division par zéro

        // Application
        for (size_t i = 0; i < n; ++i)
            data[i] = (data[i] - vol_mean) / vol_std;

        normalized = true;

        if (verbose) {
            std::cout << "✅ Normalisation z-score appliquée"
                      << " (mean=" << std::fixed << std::setprecision(4) << vol_mean
                      << ", std=" << vol_std << ")" << std::endl;
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    static std::string splitPrefix(Split s) {
        switch (s) {
            case Split::TRAIN: return "train";
            case Split::VAL:   return "val";
            case Split::TEST:  return "test";
        }
        return "train";
    }

    static std::string splitName(Split s) {
        switch (s) {
            case Split::TRAIN: return "Train";
            case Split::VAL:   return "Validation";
            case Split::TEST:  return "Test";
        }
        return "Train";
    }
};