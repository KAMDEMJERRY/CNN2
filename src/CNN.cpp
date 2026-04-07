#include "CNN.hpp"
#include "shared.hpp"
#include "IDataLoader.hpp"
#include "Layer.hpp"
#include "LossLayer.hpp"
#include "Optimizer.hpp"
#include "Tensor.hpp"
#include "ModelSerializer.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

// =============================================================================
// Construction
// =============================================================================

void CNN::addLayer(std::shared_ptr<Layer> layer) { layers_.push_back(std::move(layer)); }
void CNN::setLossLayer(std::shared_ptr<LossLayer> loss) { loss_layer_ = std::move(loss); }
void CNN::setOptimizer(std::shared_ptr<Optimizer> opt) { optimizer_ = std::move(opt); }

// =============================================================================
// Accesseurs
// =============================================================================

Layer* CNN::getLayer(int idx) const { return layers_.at(idx).get(); }
std::shared_ptr<LossLayer>                 CNN::getLossLayer()    const { return loss_layer_; }
const std::vector<std::shared_ptr<Layer>>& CNN::getLayers()       const { return layers_; }

// =============================================================================
// Passes
// =============================================================================

Tensor CNN::forward(const Tensor& input) {
    Tensor output = input;
    for (const auto& layer : layers_) {
        // std::cout<< layer->getName() << std::endl;
        output = layer->forward(output);
    }
    return output;
}

Tensor CNN::backward(const Tensor& grad_output) {
    Tensor grad = grad_output;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
        grad = (*it)->backward(grad);
    return grad;
}

void CNN::updateWeights() {
    requireOptimizer();
    for (auto& layer : layers_)
        if (layer->isTrainable) layer->updateParams(*optimizer_);
}

// =============================================================================
// Entraînement — IDataLoader (DataLoader ET DataLoader3D)
// =============================================================================

void CNN::fit(IDataLoader& dataloader, int epochs, int /*batch_size*/) {
    requireLoss();
    for (int epoch = 0; epoch < epochs; ++epoch) {
        EpochMetrics m = runEpoch(dataloader, /*train=*/true);
        printEpochStats(epoch, epochs, m);
    }
}

void CNN::fitWithValidation(IDataLoader& train_loader, IDataLoader& val_loader,
    int epochs, int /*batch_size*/) {
    requireLoss();
    float best_val_loss = std::numeric_limits<float>::max();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        EpochMetrics train_m = runEpoch(train_loader, /*train=*/true);
        EpochMetrics val_m = runEpoch(val_loader,   /*train=*/false);

        if (val_m.loss < best_val_loss) best_val_loss = val_m.loss;

        printEpochStats(epoch, epochs, train_m, &val_m);
        logEpochStats(epoch, epochs, train_m, &val_m);
    }
}

// =============================================================================
// Entraînement — Tensor
// =============================================================================

void CNN::fit(const Tensor& inputs, const Tensor& targets, int epochs, int batch_size) {
    requireLoss();
    for (int epoch = 0; epoch < epochs; ++epoch) {
        EpochMetrics m = runEpoch(inputs, targets, batch_size, /*train=*/true);
        printEpochStats(epoch, epochs, m);
    }
}

// =============================================================================
// Évaluation / Inférence
// =============================================================================
void CNN::evaluate(IDataLoader& loader) {
    EpochMetrics test_m = runEpoch(loader,   /*train=*/false);
    printTestStats(test_m);
}


float CNN::evaluate(const Tensor& inputs, const Tensor& targets) {
    requireLoss();
    Tensor predictions = forward(inputs);
    loss_layer_->setTargets(targets);
    loss_layer_->forward(predictions);
    return loss_layer_->getCurrentLoss();
}

Tensor CNN::predict(const Tensor& inputs) { return forward(inputs); }

// =============================================================================
// Sérialisation des paramètres (Boost)
// =============================================================================

void CNN::saveParameters(const std::string& filename) const {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("[CNN] Impossible d'ouvrir le fichier pour l'écriture: " + filename);
    }
    boost::archive::binary_oarchive oa(ofs);
    
    int num_layers = layers_.size();
    oa << num_layers;
    
    for (const auto& layer : layers_) {
        layer->saveParameters(oa);
    }
    std::cout << "[CNN] Paramètres sauvegardés dans: " << filename << std::endl;
}

void CNN::loadParameters(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("[CNN] Impossible d'ouvrir le fichier pour la lecture: " + filename);
    }
    boost::archive::binary_iarchive ia(ifs);
    
    int num_layers;
    ia >> num_layers;
    
    if (num_layers != static_cast<int>(layers_.size())) {
        throw std::runtime_error("[CNN] Le modèle lu a " + std::to_string(num_layers) + 
            " couches, mais le CNN courant en a " + std::to_string(layers_.size()) + ".");
    }
    
    for (const auto& layer : layers_) {
        layer->loadParameters(ia);
    }
    std::cout << "[CNN] Paramètres chargés depuis: " << filename << std::endl;
}

// =============================================================================
// Benchmarking
// =============================================================================

double CNN::benchmarkForward(const Tensor& input, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) forward(input);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count() / iterations;
}

// =============================================================================
// Boucle d'epoch — IDataLoader
// =============================================================================

EpochMetrics CNN::runEpoch(IDataLoader& loader, bool train) {
    SoftmaxCrossEntropyLayer* softmax_ce = findSoftmaxCELayer();
    loader.reset();

    float total_loss = 0.0f;
    float total_acc = 0.0f;
    int   batches = 0;
    auto  t_start = std::chrono::high_resolution_clock::now();

    while (loader.hasNext()) {
        auto [images, targets] = loader.nextBatch();
        Tensor predictions = forward(images);
        float  loss = 0.0f;
        if (softmax_ce) {
            softmax_ce->setTargets(targets);
            loss = softmax_ce->getCurrentLoss();
            if (train) {
                backward(softmax_ce->backward(Tensor()));
                updateWeights();
            }
        }
        else {
            loss_layer_->setTargets(targets);
            loss_layer_->forward(predictions);
            loss = loss_layer_->getCurrentLoss();
            if (train) {
                backward(loss_layer_->backward(Tensor()));
                updateWeights();
            }
        }

        total_loss += loss;
        total_acc += loader.computeAccuracy(predictions, targets);
        ++batches;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    return { total_loss / batches,
             total_acc / batches,
             std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() };
}

// =============================================================================
// Boucle d'epoch — Tensor
// =============================================================================

EpochMetrics CNN::runEpoch(const Tensor& inputs, const Tensor& targets,
    int batch_size, bool train) {
    SoftmaxCrossEntropyLayer* softmax_ce = findSoftmaxCELayer();

    const int num_samples = inputs.dim(0);
    const int num_batches = (num_samples + batch_size - 1) / batch_size;

    float total_loss = 0.0f;
    int   total_correct = 0;
    auto  t_start = std::chrono::high_resolution_clock::now();

    for (int b = 0; b < num_batches; ++b) {
        Tensor batch_in = extractBatch(inputs, b, batch_size);
        Tensor batch_tgt = extractBatch(targets, b, batch_size);
        const int n = batch_in.dim(0);

        Tensor output = forward(batch_in);
        float  loss = 0.0f;
        float  acc = 0.0f;

        if (softmax_ce) {
            softmax_ce->setTargets(batch_tgt);
            loss = softmax_ce->getCurrentLoss();
            acc = softmax_ce->computeAccuracy();
            if (train) { backward(softmax_ce->backward(Tensor())); updateWeights(); }
        }
        else {
            loss_layer_->setTargets(batch_tgt);
            loss_layer_->forward(output);
            loss = loss_layer_->getCurrentLoss();
            if (train) { backward(loss_layer_->backward(Tensor())); updateWeights(); }
        }

        total_loss += loss * n;
        total_correct += static_cast<int>(acc * n);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    return { total_loss / num_samples,
             static_cast<float>(total_correct) / num_samples,
             std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() };
}

// =============================================================================
// Utilitaires privés
// =============================================================================

SoftmaxCrossEntropyLayer* CNN::findSoftmaxCELayer() const {
    for (const auto& layer : layers_)
        if (auto* p = dynamic_cast<SoftmaxCrossEntropyLayer*>(layer.get())) return p;
    return nullptr;
}

Tensor CNN::extractBatch(const Tensor& data, int batch_idx, int batch_size) {
    const int start = batch_idx * batch_size;
    const int end = std::min(start + batch_size, data.dim(0));
    const int n = end - start;

    if (data.ndim() == 4) {
        Tensor batch(n, data.dim(1), data.dim(2), data.dim(3));
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < data.dim(1); ++c)
                for (int h = 0; h < data.dim(2); ++h)
                    for (int w = 0; w < data.dim(3); ++w)
                        batch(i, c, h, w) = data(start + i, c, h, w);
        return batch;
    }

    Tensor batch(n, data.dim(1), data.dim(2), data.dim(3), data.dim(4));
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < data.dim(1); ++c)
            for (int d = 0; d < data.dim(2); ++d)
                for (int h = 0; h < data.dim(3); ++h)
                    for (int w = 0; w < data.dim(4); ++w)
                        batch(i, c, d, h, w) = data(start + i, c, d, h, w);
    return batch;
}

void CNN::printEpochStats(int epoch, int total_epochs,
    const EpochMetrics& train, const EpochMetrics* val) {
    std::cout << "Epoch " << std::setw(3) << epoch + 1 << "/" << total_epochs
        << " | Loss: " << std::fixed << std::setprecision(4) << train.loss
        << " | Acc: " << std::setprecision(2) << train.accuracy * 100.0f << "%"
        << " | Time: " << train.ms << "ms";
    if (val)
        std::cout << " | Val Loss: " << std::setprecision(4) << val->loss
        << " | Val Acc: " << std::setprecision(2) << val->accuracy * 100.0f << "%";
    std::cout << "\n";

}

void CNN::printTestStats(const EpochMetrics& test) {
    std::cout << "Test "
        << " | Loss: " << std::fixed << std::setprecision(4) << test.loss
        << " | Acc: " << std::setprecision(2) << test.accuracy * 100.0f << "%"
        << " | Time: " << test.ms << "ms";
    std::cout << "\n";
}


void logEpochStats(int epoch, int total_epochs, const EpochMetrics& train, const EpochMetrics* val) {
    std::string logfile = relativePath("/logs/training_log.txt");
    std::ofstream log_file(logfile, std::ios::app);
    if (epoch == 0) {
        std::string now = currentTime();
        log_file << std::endl << now << std::string(50, '=') << std::endl;
    }
    log_file << "Epoch " << std::setw(3) << epoch + 1 << "/" << total_epochs
        << " | Loss: " << std::fixed << std::setprecision(4) << train.loss
        << " | Acc: " << std::setprecision(2) << train.accuracy * 100.0f << "%";
    log_file << " | Time: " << train.ms << "ms";
    if (val)log_file << " | Val Loss: " << std::setprecision(4) << val->loss
        << " | Val Acc: " << std::setprecision(2) << val->accuracy * 100.0f << "%";
    log_file << "\n";

    log_file.close();

}

void CNN::requireOptimizer() const {
    if (!optimizer_) throw std::runtime_error("CNN: aucun optimizer défini (setOptimizer())");
}

void CNN::requireLoss() const {
    if (!loss_layer_ && !findSoftmaxCELayer())
        throw std::runtime_error("CNN: aucune fonction de perte définie");
}