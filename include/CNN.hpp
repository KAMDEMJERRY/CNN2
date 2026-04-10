#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <Eigen/Dense>
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
// Early Stopping
// =============================================================================
struct EarlyStopping {
    int   patience  = 20;       // nb d'époques sans amélioration avant arrêt
    float min_delta = 1e-4f;    // amélioration minimale requise pour réinitialiser le compteur
    bool  restore_best = true;  // restaurer les poids du meilleur modèle à la fin
    std::string checkpoint = "./models/early_stop_best.bin"; // chemin de sauvegarde

    // Retourne true si l'entraînement doit s'arrêter
    // Appelé avec la val_loss de l'époque courante
    bool step(float val_loss) {
        if (val_loss < best_loss_ - min_delta) {
            best_loss_   = val_loss;
            wait_        = 0;
            improved_    = true;
        } else {
            ++wait_;
            improved_ = false;
        }
        return wait_ >= patience;
    }

    bool  improved()   const { return improved_; }
    float best_loss()  const { return best_loss_; }
    int   wait_count() const { return wait_; }

private:
    float best_loss_ = std::numeric_limits<float>::max();
    int   wait_      = 0;
    bool  improved_  = false;
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

    // Sans early stopping
    void fitWithValidation(IDataLoader& train_loader,
        IDataLoader& val_loader,
        int epochs,
        int batch_size);

    // Avec early stopping
    void fitWithValidation(IDataLoader& train_loader,
        IDataLoader& val_loader,
        int epochs,
        int batch_size,
        EarlyStopping& es);


    // --- Évaluation / Inférence ---
    void evaluate(IDataLoader& loader);
    float  evaluate(const Tensor& inputs, const Tensor& targets);
    Tensor predict(const Tensor& inputs);

    // --- Sérialisation des paramètres (Boost) ---
    void saveParameters(const std::string& filename) const;
    void loadParameters(const std::string& filename);

    // --- Benchmarking ---
    double benchmarkForward(const Tensor& input, int iterations = 100);


    // Matrice de confusion — inférence uniquement, pas de gradient
    // Retourne une matrice (num_classes × num_classes)
    // confusion[pred][true] — ligne = prédit, colonne = réel
    Eigen::MatrixXi confusionMatrix(IDataLoader& loader, int num_classes);
    void printConfusionMatrix(const Eigen::MatrixXi& cm,
                          const std::vector<std::string>& class_names = {});

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
