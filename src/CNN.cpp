#include "CNN.hpp"

// Ajustez ces includes selon votre structure de projet
// Si tout est dans Layer.hpp, gardez seulement ces lignes :
#include "Layer.hpp"
#include "LossLayer.hpp"
#include "Optimizer.hpp"
#include "DataLoader.hpp"
#include "Tensor.hpp"

// Si vous avez des fichiers séparés, décommentez :
// #include "DenseLayer.hpp"
// #include "ConvLayer.hpp"
// #include "SoftmaxCrossEntropyLayer.hpp"

// ============================================================================
// Implémentation de CNN
// ============================================================================

Layer* CNN::getLayer(int idx) {
    return layers[idx].get();
}

void CNN::addLayer(const std::shared_ptr<Layer>& layer) {
    layers.push_back(layer);
}

void CNN::setLossLayer(const std::shared_ptr<LossLayer>& loss) {
    loss_layer = loss;
}

std::shared_ptr<LossLayer> CNN::getLossLayer() {
    return loss_layer;
}

std::vector<std::shared_ptr<Layer>> CNN::getLayers() {
    return layers;
}

void CNN::setOptimizer(const std::shared_ptr<Optimizer>& opt) {
    optimizer = opt;
}

Tensor CNN::forward(const Tensor& input) {
    Tensor output = input;
    for (const auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

Tensor CNN::backward(const Tensor& gradOutput) {
    Tensor grad = gradOutput;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = (*it)->backward(grad);
    }
    return grad;
}

void CNN::fit(DataLoader& dataloader, int epochs, int batch_size) {
    SoftmaxCrossEntropyLayer* softmax_ce_layer = findSoftmaxCELayer();

    if (softmax_ce_layer) {
        fitWithIntegratedLoss(dataloader, epochs, batch_size, softmax_ce_layer);
    }
    else if (loss_layer) {
        fitWithSeparateLoss(dataloader, epochs, batch_size);
    }
    else {
        throw std::runtime_error("Aucune fonction de perte définie!");
    }
}

void CNN::fit(const Tensor& inputs, const Tensor& targets,
    int epochs, int batch_size) {

    SoftmaxCrossEntropyLayer* softmax_ce_layer = findSoftmaxCELayer();

    if (softmax_ce_layer) {
        fitWithIntegratedLoss(inputs, targets, epochs, batch_size, softmax_ce_layer);
    }
    else if (loss_layer) {
        fitWithSeparateLoss(inputs, targets, epochs, batch_size);
    }
    else {
        throw std::runtime_error("Aucune fonction de perte définie!");
    }
}

float CNN::evaluate(const Tensor& inputs, const Tensor& targets) {
    if (!loss_layer) {
        throw std::runtime_error("Loss layer not set!");
    }

    Tensor predictions = forward(inputs);
    loss_layer->setTargets(targets);
    loss_layer->forward(predictions);

    return loss_layer->getCurrentLoss();
}

Tensor CNN::predict(const Tensor& inputs) {
    return forward(inputs);
}

double CNN::benchmarkForward(const Tensor& input, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        forward(input);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    return duration.count() / iterations;
}

void CNN::updateWeights() {
    for (auto& layer : layers) {
        if (layer->isTrainable) {
            layer->updateParams(*optimizer);
        }
    }
}


void CNN::fitWithValidation(DataLoader& train_loader, DataLoader& val_loader,
    int epochs, int batch_size) {

    float best_val_loss = std::numeric_limits<float>::max();
    int best_epoch = 0;


    for (int epoch = 0; epoch < epochs; ++epoch) {
        // ========== PHASE D'ENTRAÎNEMENT ==========
        train_loader.reset();
        float train_loss = 0.0f;
        float train_acc = 0.0f;
        int train_batches = 0;

        auto train_start = std::chrono::high_resolution_clock::now();

        while (train_loader.hasNext()) {
            auto [images, targets] = train_loader.nextBatch();
            Tensor predictions = forward(images);


            std::cout << "Set targets \n" << std::endl;
            loss_layer->setTargets(targets);
            loss_layer->forward(predictions);
            float loss = loss_layer->getCurrentLoss();

            Tensor grad = loss_layer->backward(Tensor());
            backward(grad);
            updateWeights();

            train_loss += loss;
            train_acc += train_loader.computeAccuracy(predictions, targets);
            train_batches++;
        }

        auto train_end = std::chrono::high_resolution_clock::now();
        auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            train_end - train_start);

        // ========== PHASE DE VALIDATION ==========
        val_loader.reset();
        float val_loss = 0.0f;
        float val_acc = 0.0f;
        int val_batches = 0;

        while (val_loader.hasNext()) {
            auto [images, targets] = val_loader.nextBatch();
            Tensor predictions = forward(images);

            loss_layer->setTargets(targets);
            loss_layer->forward(predictions);
            val_loss += loss_layer->getCurrentLoss();
            val_acc += val_loader.computeAccuracy(predictions, targets);
            val_batches++;
        }

        // ========== CALCUL DES MÉTRIQUES MOYENNES ==========
        float mean_train_loss = train_loss / train_batches;
        float mean_train_acc = (train_acc / train_batches) * 100;
        float mean_val_loss = val_loss / val_batches;
        float mean_val_acc = (val_acc / val_batches) * 100;

        // ========== AFFICHAGE ==========
        std::cout << "Epoch " << std::setw(3) << epoch + 1 << "/" << epochs
            << " | Train Loss: " << std::fixed << std::setprecision(4) << mean_train_loss
            << " | Train Precision: " << std::setprecision(2) << mean_train_acc << "%"
            << " | Val Loss: " << std::setprecision(4) << mean_val_loss << std::endl;
            // << " | Val Acc: " << std::setprecision(2) << mean_val_acc << "%"
            // << " | Time: " << train_time.count() << "ms";

        // Marquer le meilleur modèle
        // if (mean_val_loss < best_val_loss) {
        //     best_val_loss = mean_val_loss;
        //     best_epoch = epoch + 1;
        //     std::cout << " 🏆";
        // }
        // std::cout << std::endl;
    }

    // std::cout << "\n" << std::string(70, '=') << std::endl;
    // std::cout << "✅ ENTRAÎNEMENT TERMINÉ" << std::endl;
    // std::cout << "   Meilleure validation loss: " << std::fixed << std::setprecision(4) 
    //           << best_val_loss << " (epoch " << best_epoch << ")" << std::endl;
    // std::cout << std::string(70, '=') << "\n" << std::endl;
}

SoftmaxCrossEntropyLayer* CNN::findSoftmaxCELayer() {
    for (auto& layer : layers) {
        if (auto softmax_ce = dynamic_cast<SoftmaxCrossEntropyLayer*>(layer.get())) {
            return softmax_ce;
        }
    }
    return nullptr;
}

void CNN::fitWithSeparateLoss(DataLoader& dataloader,
    int epochs, int batch_size) {

    for (int epoch = 0; epoch < epochs; ++epoch) {
        dataloader.reset();
        float epoch_loss = 0.0f;
        float epoch_acc = 0.0f;
        int num_batches = 0;

        while (dataloader.hasNext()) {
            auto [images, targets] = dataloader.nextBatch();

            // Forward
            Tensor predictions = forward(images);

            // Loss
            getLossLayer()->setTargets(targets);
            getLossLayer()->forward(predictions);
            float loss = getLossLayer()->getCurrentLoss();

            // Backward
            Tensor grad = getLossLayer()->backward(Tensor());
            backward(grad);

            // Update weights
            updateWeights();

            // Metrics
            float acc = dataloader.computeAccuracy(predictions, targets);

            epoch_loss += loss;
            epoch_acc += acc;
            num_batches++;
        }

        std::cout << "Epoch " << epoch + 1 << "/" << epochs
            << " - Loss: " << epoch_loss / num_batches
            << " - Acc: " << (epoch_acc / num_batches) * 100 << "%"
            << std::endl;
    }
}

void CNN::fitWithIntegratedLoss(DataLoader& dataloader,
    int epochs, int batch_size,
    SoftmaxCrossEntropyLayer* softmax_ce_layer) {

    for (int epoch = 0; epoch < epochs; ++epoch) {
        dataloader.reset();
        float epoch_loss = 0.0f;
        float epoch_acc = 0.0f;
        int num_batches = 0;

        while (dataloader.hasNext()) {
            auto [images, targets] = dataloader.nextBatch();

            // Forward
            Tensor predictions = forward(images);

            // Définir les cibles dans la couche intégrée
            softmax_ce_layer->setTargets(targets);

            // Calculer la perte et l'accuracy
            float loss = softmax_ce_layer->computeLoss();

            // Backward
            Tensor grad = softmax_ce_layer->backward(Tensor());
            backward(grad);

            // Update weights
            updateWeights();

            // Metrics
            float acc = dataloader.computeAccuracy(predictions, targets);

            epoch_loss += loss;
            epoch_acc += acc;
            num_batches++;
        }

        std::cout << "Epoch " << epoch + 1 << "/" << epochs
            << " - Loss: " << epoch_loss / num_batches
            << " - Acc: " << (epoch_acc / num_batches) * 100 << "%"
            << std::endl;
    }
}

void CNN::fitWithSeparateLoss(const Tensor& inputs, const Tensor& targets,
    int epochs, int batch_size) {
    int num_samples = inputs.dim(0);
    int num_batches = (num_samples + batch_size - 1) / batch_size;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;

        for (int batch = 0; batch < num_batches; ++batch) {
            // Extraire batch
            Tensor batch_input = extractBatch(inputs, batch, batch_size);
            Tensor batch_target = extractBatch(targets, batch, batch_size);

            // Forward pass à travers toutes les couches
            Tensor output = forward(batch_input);

            // Calcul de la perte avec LossLayer
            loss_layer->setTargets(batch_target);
            loss_layer->forward(output);

            float batch_loss = loss_layer->getCurrentLoss();
            epoch_loss += batch_loss * batch_input.dim(0);

            // Backward pass
            Tensor loss_grad = loss_layer->backward(Tensor());
            backward(loss_grad);

            // Mise à jour des poids
            updateWeights();
        }

        epoch_loss /= num_samples;
        std::cout << "Epoch " << epoch + 1 << " - Loss: " << epoch_loss << std::endl;
    }
}

void CNN::fitWithIntegratedLoss(const Tensor& inputs, const Tensor& targets,
    int epochs, int batch_size,
    SoftmaxCrossEntropyLayer* softmax_ce_layer) {
    int num_samples = inputs.dim(0);
    int num_batches = (num_samples + batch_size - 1) / batch_size;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int epoch_correct = 0;

        for (int batch = 0; batch < num_batches; ++batch) {
            Tensor batch_input = extractBatch(inputs, batch, batch_size);
            Tensor batch_target = extractBatch(targets, batch, batch_size);

            // Forward pass
            Tensor output = forward(batch_input);

            // Définir les cibles dans la couche intégrée
            softmax_ce_layer->setTargets(batch_target);

            // Calculer la perte et la précision
            float batch_loss = softmax_ce_layer->computeLoss();
            float batch_accuracy = softmax_ce_layer->computeAccuracy(batch_target);

            epoch_loss += batch_loss * batch_input.dim(0);
            epoch_correct += static_cast<int>(batch_accuracy * batch_input.dim(0));

            // Backward pass
            Tensor grad = softmax_ce_layer->backward(Tensor());
            backward(grad);

            // Mise à jour
            updateWeights();
        }

        epoch_loss /= num_samples;
        float epoch_accuracy = static_cast<float>(epoch_correct) / num_samples;

        std::cout << "Epoch " << epoch + 1
            << " - Loss: " << epoch_loss
            << " - Accuracy: " << epoch_accuracy * 100 << "%"
            << std::endl;
    }
}

Tensor CNN::extractBatch(const Tensor& data, int batch_idx, int batch_size) {
    int start = batch_idx * batch_size;
    int end = std::min(start + batch_size, data.dim(0));
    int current_size = end - start;

    Tensor batch(current_size, data.dim(1), data.dim(2), data.dim(3));

    // Copie des données
    for (int i = 0; i < current_size; ++i) {
        for (int c = 0; c < data.dim(1); ++c) {
            for (int h = 0; h < data.dim(2); ++h) {
                for (int w = 0; w < data.dim(3); ++w) {
                    batch(i, c, h, w) = data(start + i, c, h, w);
                }
            }
        }
    }

    return batch;
}