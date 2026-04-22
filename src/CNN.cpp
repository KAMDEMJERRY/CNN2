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
#include <filesystem>
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

        bool improved = (val_m.loss < best_val_loss);
        if (improved) best_val_loss = val_m.loss;

        printEpochStats(epoch, epochs, train_m, &val_m);
        logEpochStats(epoch, epochs, train_m, &val_m, "./logs/training_log.txt", improved);
    }
}

// ---------------------------------------------------------------------------
// fitWithValidation avec Early Stopping
// ---------------------------------------------------------------------------
//
//  Logique :
//    - Chaque époque, es.step(val_loss) vérifie si la perte de validation
//      s'améliore d'au moins min_delta.
//    - Si amélioration → sauvegarde le checkpoint (meilleur modèle).
//    - Sinon → incrémente le compteur de patience.
//    - Si patience épuisée → arrêt et restauration du checkpoint si
//      es.restore_best == true.
//
void CNN::fitWithValidation(IDataLoader& train_loader, IDataLoader& val_loader,
    int epochs, int /*batch_size*/, EarlyStopping& es) {
    requireLoss();

    // Créer le dossier models si absent
    std::filesystem::create_directories(
        std::filesystem::path(es.checkpoint).parent_path());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        EpochMetrics train_m = runEpoch(train_loader, /*train=*/true);
        EpochMetrics val_m   = runEpoch(val_loader,   /*train=*/false);

        const bool stop = es.step(val_m.loss);

        // Affichage enrichi avec indicateur d'amélioration et compteur
        printEpochStats(epoch, epochs, train_m, &val_m);

        if (es.improved()) {
            std::cout << "  ✓ Early stopping — meilleure val_loss: "
                      << std::fixed << std::setprecision(4) << es.best_loss()
                      << " → checkpoint sauvegardé\n";
            saveParameters(es.checkpoint);
        } else {
            std::cout << "  · Pas d'amélioration ("
                      << es.wait_count() << "/" << es.patience << ")\n";
        }

        logEpochStats(epoch, epochs, train_m, &val_m, es.log_file, es.improved(), es.improved() ? es.checkpoint : std::string());


        if (stop) {
            std::cout << "\n[Early Stopping] Arrêt à l'époque " << epoch + 1
                      << " — patience épuisée (" << es.patience << " époques).\n";
            if (es.restore_best && std::filesystem::exists(es.checkpoint)) {
                std::cout << "[Early Stopping] Restauration du meilleur modèle: "
                          << es.checkpoint << "\n";
                loadParameters(es.checkpoint);
            }
            break;
        }
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
        << " | Loss: " << std::fixed << std::setprecision(4) << train.loss;

    if (val)
        std::cout << " | Val Loss: " << std::setprecision(4) << val->loss;

    std::cout << " | Acc: " << std::setprecision(2) << train.accuracy * 100.0f << "%";

    if (val)
        std::cout << " | Val Acc: " << std::setprecision(2) << val->accuracy * 100.0f << "%";

    int min = train.ms/(1000 * 60);
    std::cout << " | Time: " << train.ms << "ms" << std::setprecision(2) << "("<<  min <<"min)";

    std::cout << "\n";

}

void CNN::printTestStats(const EpochMetrics& test) {
    std::cout << "Test "
        << " | Loss: " << std::fixed << std::setprecision(4) << test.loss
        << " | Acc: " << std::setprecision(2) << test.accuracy * 100.0f << "%"
        << " | Time: " << test.ms << "ms";
    std::cout << "\n";
}


void logEpochStats(int epoch, int total_epochs, const EpochMetrics& train, const EpochMetrics* val, const std::string& logfile_path, bool improved /*= false*/, const std::string& checkpoint_path /*= "" */) {
    // std::string logfile = relativePath(logfile_path);
    std::ofstream log_file(logfile_path, std::ios::app);
    if(!log_file.is_open()){
        std::cerr << "Error: Could not open log file " << logfile_path << std::endl;
        return;
    }

    if (epoch == 0) {
        std::string now = currentTime();
        log_file << std::endl << now << std::string(50, '=') << std::endl;
    }
    log_file << "Epoch " << std::setw(3) << epoch + 1 << "/" << total_epochs
        << " | Loss: " << std::fixed << std::setprecision(4) << train.loss
        << " | Acc: " << std::setprecision(2) << train.accuracy * 100.0f << "%";
    log_file << " | Time: " << train.ms << "ms";
    if (val) log_file << " | Val Loss: " << std::setprecision(4) << val->loss
        << " | Val Acc: " << std::setprecision(2) << val->accuracy * 100.0f << "%";

    // Marquer les améliorations / checkpoints dans le fichier de log
    if (improved) {
        log_file << " | Improved ✓";
        if (!checkpoint_path.empty()) log_file << " [ckpt: " << checkpoint_path << "]";
    }

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



// =============================================================================
// Export CSV pour ROC-AUC
// =============================================================================

void CNN::exportPredictionsToCSV(IDataLoader& loader, const std::string& filename) {
    std::filesystem::path p(filename);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }
    
    std::ofstream ofs(filename);
    if (!ofs) {
        throw std::runtime_error("[CNN] Impossible d'ouvrir " + filename + " pour export CSV");
    }

    loader.reset();
    
    // Obtenir le nombre de classes à partir de la première prédiction
    bool header_written = false;

    while (loader.hasNext()) {
        auto [images, targets] = loader.nextBatch();
        Tensor preds = forward(images);
        const int B = preds.dim(0);
        const int C = preds.dim(1);

        if (!header_written) {
            ofs << "true_label";
            for (int c = 0; c < C; ++c) {
                ofs << ",prob_class_" << c;
            }
            ofs << "\n";
            header_written = true;
        }

        for (int b = 0; b < B; ++b) {
            // Classe vraie : index scalaire dans targets
            int true_class = 0;
            if (targets.dim(1) == 1) {
                // Stockage comme index scalaire
                true_class = static_cast<int>(targets(b, 0, 0, 0, 0));
            } else {
                // Stockage one-hot → argmax
                float max_t = targets(b, 0, 0, 0, 0);
                for (int c = 1; c < targets.dim(1); ++c) {
                    const float v = targets(b, c, 0, 0, 0);
                    if (v > max_t) { max_t = v; true_class = c; }
                }
            }

            ofs << true_class;

            // Comme softmax est potentiellement dans LossLayer, nous prenons les sorties
            // Si Model a une Softmax indépendante, ce sont les probs.
            // S'il n'y en a pas, c'est des logits. Pour ROC AUC, l'ordre des logits 
            // préserve l'AUC de toute façon, mais on peut appliquer softmax par précaution.
            
            // Calcul du sum exp pour Softmax instable...
            float max_val = preds(b, 0, 0, 0, 0);
            for (int c = 1; c < C; ++c) {
                if (preds(b, c, 0, 0, 0) > max_val) max_val = preds(b, c, 0, 0, 0);
            }
            float sum_exp = 0.0f;
            for (int c = 0; c < C; ++c) {
                sum_exp += std::exp(preds(b, c, 0, 0, 0) - max_val);
            }

            for (int c = 0; c < C; ++c) {
                float prob = std::exp(preds(b, c, 0, 0, 0) - max_val) / sum_exp;
                ofs << "," << prob;
            }
            ofs << "\n";
        }
    }
    
    std::cout << "[CNN] Prédictions exportées dans: " << filename << std::endl;
}

// =============================================================================
// Matrice de confusion
// =============================================================================

Eigen::MatrixXi CNN::confusionMatrix(IDataLoader& loader, int num_classes) {
    Eigen::MatrixXi cm = Eigen::MatrixXi::Zero(num_classes, num_classes);
    loader.reset();

    while (loader.hasNext()) {
        auto [images, targets] = loader.nextBatch();
        Tensor preds = forward(images);
        const int B = preds.dim(0);
        const int C = preds.dim(1);

        for (int b = 0; b < B; ++b) {
            // Classe prédite : argmax sur les canaux
            int pred_class = 0;
            float max_val  = preds(b, 0, 0, 0, 0);
            for (int c = 1; c < C; ++c) {
                const float v = preds(b, c, 0, 0, 0);
                if (v > max_val) { max_val = v; pred_class = c; }
            }

            // Classe vraie : index scalaire dans targets
            // targets est (B, 1, 1, 1, 1) ou (B, num_classes, 1, 1, 1)
            int true_class = 0;
            if (targets.dim(1) == 1) {
                // Stockage comme index scalaire
                true_class = static_cast<int>(targets(b, 0, 0, 0, 0));
            } else {
                // Stockage one-hot → argmax
                float max_t = targets(b, 0, 0, 0, 0);
                for (int c = 1; c < targets.dim(1); ++c) {
                    const float v = targets(b, c, 0, 0, 0);
                    if (v > max_t) { max_t = v; true_class = c; }
                }
            }

            if (true_class >= 0 && true_class < num_classes &&
                pred_class >= 0 && pred_class < num_classes)
                cm(true_class, pred_class)++;
        }
    }
    return cm;
}

void CNN::printConfusionMatrix(const Eigen::MatrixXi& cm,
                                const std::vector<std::string>& class_names) {
    const int N = cm.rows();
    std::cout << "\n── Matrice de confusion ─────────────────────────────\n";

    // En-tête colonnes (prédictions)
    std::cout << std::setw(12) << " ";
    for (int j = 0; j < N; ++j)
        std::cout << std::setw(10)
                  << (class_names.empty() ? "Pred_" + std::to_string(j)
                                          : class_names[j]);
    std::cout << "\n";

    for (int i = 0; i < N; ++i) {
        std::cout << std::setw(12)
                  << (class_names.empty() ? "True_" + std::to_string(i)
                                          : class_names[i]);
        for (int j = 0; j < N; ++j) {
            // Diagonale en vert, erreurs en rouge
            if (i == j)
                std::cout << green << std::setw(10) << cm(i, j) << reset;
            else if (cm(i, j) > 0)
                std::cout << red << std::setw(10) << cm(i, j) << reset;
            else
                std::cout << std::setw(10) << cm(i, j);
        }
        std::cout << "\n";
    }

    // Métriques par classe
    std::cout << "\n── Métriques par classe ─────────────────────────────\n";
    std::cout << std::setw(12) << " "
              << std::setw(10) << "Precision"
              << std::setw(10) << "Recall"
              << std::setw(10) << "F1"
              << std::setw(10) << "Support"
              << "\n";

    float macro_f1 = 0.f;
    for (int i = 0; i < N; ++i) {
        int tp = cm(i, i);
        int fp = cm.col(i).sum() - tp;  // prédit i mais vrai != i
        int fn = cm.row(i).sum() - tp;  // vrai i mais prédit != i

        float precision = (tp + fp > 0) ? (float)tp / (tp + fp) : 0.f;
        float recall    = (tp + fn > 0) ? (float)tp / (tp + fn) : 0.f;
        float f1        = (precision + recall > 0)
                          ? 2.f * precision * recall / (precision + recall) : 0.f;
        int   support   = cm.row(i).sum();
        macro_f1 += f1;

        std::cout << std::setw(12)
                  << (class_names.empty() ? "Classe_" + std::to_string(i)
                                          : class_names[i])
                  << std::fixed << std::setprecision(3)
                  << std::setw(10) << precision
                  << std::setw(10) << recall
                  << std::setw(10) << f1
                  << std::setw(10) << support
                  << "\n";
    }
    std::cout << "\n   Macro F1 : " << macro_f1 / N << "\n";
    std::cout << "─────────────────────────────────────────────────────\n\n";
}