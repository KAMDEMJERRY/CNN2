#pragma once
#include "IDataLoader.hpp"
#include "MedMNIST3DDataset.hpp"
#include "Augmentation3D.hpp"
#include <algorithm>
#include <numeric>
#include <optional>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>

class DataLoader3D : public IDataLoader {
public:
    DataLoader3D(MedMNIST3DDataset& dataset,
                 int          batch_size = 16,
                 bool         shuffle    = true,
                 unsigned int seed       = 42)
        : dataset_(dataset), batch_size_(batch_size),
          shuffle_(shuffle), rng_(seed)
    {
        reset();
    }

    void reset() override {
        int n = dataset_.getNumSamples();
        if (max_samples_ > 0 && max_samples_ < n) n = max_samples_;
        indices_.resize(n);
        std::iota(indices_.begin(), indices_.end(), 0);
        if (shuffle_) std::shuffle(indices_.begin(), indices_.end(), rng_);
        current_idx_ = 0;
    }

    bool hasNext() const override {
        return current_idx_ < static_cast<int>(indices_.size());
    }

    // ── nextBatch : récupère le batch puis augmente si activé ────────────────
    std::pair<Tensor, Tensor> nextBatch() override {
        if (!hasNext()) reset();
        int end = std::min(current_idx_ + batch_size_,
                           static_cast<int>(indices_.size()));
        std::vector<int> batch_indices(indices_.begin() + current_idx_,
                                       indices_.begin() + end);
        current_idx_ = end;

        auto [images, labels] = dataset_.getBatch(batch_indices);

        if (augmentor_) {
            const int B = images.dim(0);
            const int C = images.dim(1);
            const int D = images.dim(2);
            const int H = images.dim(3);
            const int W = images.dim(4);
            const int vol_size = C * D * H * W;

            for (int b = 0; b < B; ++b) {
                // Extraire le volume b : Tensor(1, C, D, H, W)
                Tensor vol(1, C, D, H, W);
                const float* src = images.getData() + b * vol_size;
                std::memcpy(vol.getData(), src, vol_size * sizeof(float));

                // Appliquer l'augmentation
                Tensor aug_vol = augmentor_->apply(vol, rng_);

                // Remettre dans le batch
                // Note : la rotation peut changer H/W si H≠W,
                // mais sur FractureMNIST3D H==W==D==28 donc la shape est stable.
                float* dst = images.getData() + b * vol_size;
                std::memcpy(dst, aug_vol.getData(), vol_size * sizeof(float));
            }
        }

        return { images, labels };
    }

    // ── Activation de l'augmentation ─────────────────────────────────────────
    // Appeler UNIQUEMENT sur le train_loader. Ne pas appeler sur val/test.
    void setAugmentation(const AugmentConfig& cfg = AugmentConfig{}) {
        augmentor_ = Augmentor3D(cfg);
    }

    // Désactive l'augmentation
    void clearAugmentation() {
        augmentor_.reset();
    }

    bool isAugmented() const { return augmentor_.has_value(); }

    float computeAccuracy(const Tensor& predictions,
                          const Tensor& targets) const override {
        return dataset_.computeAccuracy(predictions, targets);
    }

    int getNumBatches() const override {
        return (getNumSamples() + batch_size_ - 1) / batch_size_;
    }

    int getBatchSize()  const override { return batch_size_; }

    int getNumSamples() const override {
        int n = dataset_.getNumSamples();
        return (max_samples_ > 0 && max_samples_ < n) ? max_samples_ : n;
    }

    void setMaxSamples(int max_samples) override {
        max_samples_ = max_samples;
        reset();
    }

    // Spécifique à DataLoader3D
    float getProgress() const {
        return static_cast<float>(current_idx_) / dataset_.getNumSamples();
    }

    void printProgress(int epoch, int total_epochs) const {
        const int   batches_done  = current_idx_ / batch_size_;
        const int   total_batches = getNumBatches();
        const float pct           = 100.0f * batches_done / total_batches;

        std::cout << "\r  Epoch [" << epoch << "/" << total_epochs << "]"
                  << "  Batch [" << batches_done << "/" << total_batches << "]"
                  << "  (" << std::fixed << std::setprecision(1) << pct << "%)"
                  << std::flush;
    }

private:
    MedMNIST3DDataset&         dataset_;
    int                        batch_size_;
    bool                       shuffle_;
    int                        max_samples_ = -1;
    std::vector<int>           indices_;
    int                        current_idx_ = 0;
    std::mt19937               rng_;
    std::optional<Augmentor3D> augmentor_;  // actif seulement sur train_loader
};