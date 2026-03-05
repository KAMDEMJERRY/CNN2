# pragma once
# include <vector>
# include <memory>
# include <iostream>
# include <chrono>
# include <iomanip>
# include <limits>

// Forward declarations pour éviter les inclusions circulaires
class Layer;
class DenseLayer;
class ConvLayer;
class LossLayer;
class SoftmaxCrossEntropyLayer;
class Optimizer;
class DataLoader;
class Tensor;


class CNN {
private:
    std::vector<std::shared_ptr<Layer>> layers;
    std::shared_ptr<LossLayer> loss_layer;
    std::shared_ptr<Optimizer> optimizer;

public:
    CNN() = default;

    Layer* getLayer(int idx);

    void addLayer(const std::shared_ptr<Layer>& layer);

    void setLossLayer(const std::shared_ptr<LossLayer>& loss);

    std::shared_ptr<LossLayer> getLossLayer();

    std::vector<std::shared_ptr<Layer>> getLayers();

    void setOptimizer(const std::shared_ptr<Optimizer>& opt);

    Tensor forward(const Tensor& input);

    Tensor backward(const Tensor& gradOutput);

    // Entraînement simple
    void fit(DataLoader& dataloader, int epochs = 10, int batch_size = 32);

    void fit(const Tensor& inputs, const Tensor& targets,
        int epochs = 10, int batch_size = 32);

    // Entraînement avec validation
    void fitWithValidation(DataLoader& train_loader, DataLoader& val_loader,
        int epochs, int batch_size);

    // Fonction pour evaluer le modele
    float evaluate(const Tensor& inputs, const Tensor& targets);

    Tensor predict(const Tensor& inputs);

    // Benchmarking
    double benchmarkForward(const Tensor& input, int iterations = 100);

    void updateWeights();

private:

    // Pour gerer SoftmaxCrossEntropyLayer integre
    SoftmaxCrossEntropyLayer* findSoftmaxCELayer();

    void fitWithSeparateLoss(DataLoader& dataloader,
        int epochs, int batch_size);

    void fitWithIntegratedLoss(DataLoader& dataloader,
        int epochs, int batch_size,
        SoftmaxCrossEntropyLayer* softmax_ce_layer);

    void fitWithSeparateLoss(const Tensor& inputs, const Tensor& targets,
        int epochs, int batch_size);

    void fitWithIntegratedLoss(const Tensor& inputs, const Tensor& targets,
        int epochs, int batch_size,
        SoftmaxCrossEntropyLayer* softmax_ce_layer);

    Tensor extractBatch(const Tensor& data, int batch_idx, int batch_size);
};