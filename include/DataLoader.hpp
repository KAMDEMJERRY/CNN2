#pragma once
#include "ImageLoader.hpp"
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

class DataLoader {
private:
    ImageFolderDataset& dataset;
    int batch_size;
    bool shuffle;
    int num_workers;
    
    std::vector<int> indices;
    int current_idx = 0;
    std::mt19937 rng;
    
public:
    ImageLoader& getImageLoader() {
        return dataset.getImageLoader();
    }
    DataLoader(ImageFolderDataset& dataset, 
               int batch_size = 32, 
               bool shuffle = true,
               int num_workers = 1) 
        : dataset(dataset), batch_size(batch_size), 
          shuffle(shuffle), num_workers(num_workers) {
        
        reset();
    }
    
    void reset() {
        indices.resize(dataset.getNumSamples());
        for (int i = 0; i < indices.size(); ++i) indices[i] = i;
        
        if (shuffle) {
            std::shuffle(indices.begin(), indices.end(), rng);
        }
        
        current_idx = 0;
    }
    
    bool hasNext() const {
        return current_idx < indices.size();
    }
    
    std::pair<Tensor, Tensor> nextBatch() {
        if (!hasNext()) {
            reset();
        }
        
        int end = std::min(current_idx + batch_size, (int)indices.size());
        std::vector<int> batch_indices(indices.begin() + current_idx, 
                                       indices.begin() + end);
        current_idx = end;
        
        return dataset.getBatch(batch_indices);
    }
    
    int getNumBatches() const {
        return (indices.size() + batch_size - 1) / batch_size;
    }

    float computeAccuracy(Tensor& predictions, Tensor& targets){
        return dataset.computeAccuracy(predictions, targets);
    }
};