#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <limits>
#include <stdexcept>

class Layer;
class LossLayer;
class SoftmaxCrossEntropyLayer;
class Optimizer;
class IDataLoader;   // ← interface commune DataLoader / DataLoader3D
class Tensor;

// =============================================================================
// Métriques d'une epoch
// =============================================================================
struct EpochMetrics {
    float loss = 0.0f;
    float accuracy = 0.0f;
    long  ms = 0;
};

// =============================================================================
// CNN
// =============================================================================
class CNN {
public:
    CNN() = default;

    // --- Construction ---
    void addLayer(std::shared_ptr<Layer> layer);
    void setLossLayer(std::shared_ptr<LossLayer> loss);
    void setOptimizer(std::shared_ptr<Optimizer> opt);

    // --- Accesseurs ---
    Layer* getLayer(int idx) const;
    std::shared_ptr<LossLayer>                 getLossLayer()    const;
    const std::vector<std::shared_ptr<Layer>>& getLayers()       const;

    // --- Passes ---
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);
    void   updateWeights();

    // --- Entraînement (DataLoader ou DataLoader3D via IDataLoader) ---
    void fit(IDataLoader& dataloader,
        int epochs = 10,
        int batch_size = 32);

    void fit(const Tensor& inputs,
        const Tensor& targets,
        int epochs = 10,
        int batch_size = 32);

    void fitWithValidation(IDataLoader& train_loader,
        IDataLoader& val_loader,
        int epochs,
        int batch_size);


    // --- Évaluation / Inférence ---
    void evaluate(IDataLoader& loader);
    float  evaluate(const Tensor& inputs, const Tensor& targets);
    Tensor predict(const Tensor& inputs);

    // --- Benchmarking ---
    double benchmarkForward(const Tensor& input, int iterations = 100);

private:
    std::vector<std::shared_ptr<Layer>> layers_;
    std::shared_ptr<LossLayer>          loss_layer_;
    std::shared_ptr<Optimizer>          optimizer_;

    SoftmaxCrossEntropyLayer* findSoftmaxCELayer() const;

    // Boucle d'epoch commune aux deux types de DataLoader
    EpochMetrics runEpoch(IDataLoader& loader, bool train);

    // Boucle d'epoch sur Tensor
    EpochMetrics runEpoch(const Tensor& inputs, const Tensor& targets,
        int batch_size, bool train);

    static Tensor extractBatch(const Tensor& data, int batch_idx, int batch_size);

    static void printEpochStats(int epoch, int total_epochs,
        const EpochMetrics& train,
        const EpochMetrics* val = nullptr);

    void printTestStats(const EpochMetrics& test);

    void requireOptimizer() const;
    void requireLoss()      const;
};

void logEpochStats(int epoch, int total_epochs, const EpochMetrics& train, const EpochMetrics* val);
