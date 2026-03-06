#pragma once
#include "IDataLoader.hpp"
#include "ImageLoader.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

class DataLoader : public IDataLoader {
public:
    DataLoader(ImageFolderDataset& dataset,
               int  batch_size  = 32,
               bool shuffle     = true,
               int  num_workers = 1)
        : dataset_(dataset), batch_size_(batch_size),
          shuffle_(shuffle), num_workers_(num_workers)
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

    std::pair<Tensor, Tensor> nextBatch() override {
        if (!hasNext()) reset();
        int end = std::min(current_idx_ + batch_size_,
                           static_cast<int>(indices_.size()));
        std::vector<int> batch_indices(indices_.begin() + current_idx_,
                                       indices_.begin() + end);
        current_idx_ = end;
        return dataset_.getBatch(batch_indices);
    }

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

    // Spécifique à DataLoader 2D
    ImageLoader& getImageLoader() { return dataset_.getImageLoader(); }

private:
    ImageFolderDataset& dataset_;
    int                 batch_size_;
    bool                shuffle_;
    int                 num_workers_;
    int                 max_samples_ = -1;
    std::vector<int>    indices_;
    int                 current_idx_ = 0;
    std::mt19937        rng_;
};