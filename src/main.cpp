// main.cpp
#include "CNN.hpp"
#include "ConvLayer.hpp"
#include "ConvLayer3D.hpp"
#include "ActivationLayer.hpp"
#include "PoolLayer.hpp"
#include "DenseLayer.hpp"
#include "LossLayer.hpp"
#include "Optimizer.hpp"
#include "Dimensions.hpp"
#include "DropoutLayer.hpp"
#include "DataLoader.hpp"
#include "DataLoader3D.hpp"
#include "MedMNIST3DDataset.hpp"
#include <iostream>
#include <chrono>
#include <filesystem>

// =============================================================================
// Configuration globale
// =============================================================================

// Pipeline sélectionné au lancement
enum class Pipeline { CNN2D, CNN3D };
static constexpr Pipeline ACTIVE_PIPELINE = Pipeline::CNN3D;

// --- 2D (MNIST / BloodCells) ---
#define MNIST_TRAIN_PATH      "../../../dataset/mnist_img/trainingSample/trainingSample/"
#define MNIST_TEST_PATH       "../../../dataset/mnist_img/trainingSample/trainingSample/"
#define BLOODCELLS_TRAIN_PATH "../../../dataset/bloodcell/images/TRAIN/"
#define BLOODCELLS_TEST_PATH  "../../../dataset/bloodcell/images/TEST/"

static constexpr bool  USE_BLOODCELLS   = false;
static constexpr int   IMAGE_SIZE_2D    = 28;
static constexpr int   BATCH_SIZE_2D    = 100;
static constexpr int   EPOCHS_2D        = 50;
static constexpr float LR_2D            = 0.0001f;

// --- 3D (MedMNIST3D) ---
#define FRACTURE_PATH  "../../../dataset/fracturemnist3d"
#define ADRENAL_PATH   "../../../dataset/adrenalmnist3d"

static constexpr int   BATCH_SIZE_3D    = 16;
static constexpr int   EPOCHS_3D        = 30;
static constexpr float LR_3D            = 0.0001f;
static constexpr int   VOL_SIZE         = 28; // 28x28x28

// =============================================================================
// Utilitaires
// =============================================================================

static void section(const std::string& title) {
    const std::string bar(60, '=');
    std::cout << "\n" << bar << "\n  " << title << "\n" << bar << "\n";
}

static void requireDir(const std::string& path) {
    if (!std::filesystem::exists(path))
        throw std::runtime_error("Répertoire introuvable : " + path);
}

// =============================================================================
// Pipeline 2D — MNIST ou BloodCells
// =============================================================================

static CNN buildModel2D(int num_classes) {
    CNN model;

    // --- Feature extraction ---
    model.addLayer(std::make_shared<ConvLayer>(1, 32, 3, 3, 1, 1, 1, 1));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<ConvLayer>(32, 32, 3, 3, 1, 1, 1, 1));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<MaxPoolLayer>(2, 2));          // 28→14

    model.addLayer(std::make_shared<ConvLayer>(32, 64, 3, 3, 1, 1, 1, 1));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<ConvLayer>(64, 64, 3, 3, 1, 1, 1, 1));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<MaxPoolLayer>(2, 2));          // 14→7

    // --- Classification ---
    model.addLayer(std::make_shared<DenseLayer>(7 * 7 * 64, 128));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<DropoutLayer>(0.5f));
    model.addLayer(std::make_shared<DenseLayer>(128, num_classes));
    model.setLossLayer(std::make_shared<SoftmaxCrossEntropyLayer>());

    model.setOptimizer(std::make_shared<Adam>(LR_2D));
    return model;
}

static int run2D() {
    section("Pipeline 2D — " + std::string(USE_BLOODCELLS ? "BloodCells" : "MNIST"));

    const std::string train_dir = USE_BLOODCELLS ? BLOODCELLS_TRAIN_PATH : MNIST_TRAIN_PATH;
    const std::string test_dir  = USE_BLOODCELLS ? BLOODCELLS_TEST_PATH  : MNIST_TEST_PATH;

    requireDir(train_dir);
    requireDir(test_dir);

    // --- Datasets ---
    ImageFolderDataset train_dataset(train_dir, IMAGE_SIZE_2D, IMAGE_SIZE_2D, true,  true);
    ImageFolderDataset val_dataset  (test_dir,  IMAGE_SIZE_2D, IMAGE_SIZE_2D, false, true);

    const int num_classes       = train_dataset.getNumClasses();
    const int num_train_samples = train_dataset.getNumSamples();
    const int num_val_samples   = val_dataset.getNumSamples();

    std::cout << "Classes    : " << num_classes       << "\n"
              << "Train      : " << num_train_samples << "\n"
              << "Validation : " << num_val_samples   << "\n";

    if (num_train_samples == 0 || num_val_samples == 0)
        throw std::runtime_error("Dataset vide !");

    // --- Modèle ---
    section("Architecture 2D");
    CNN model = buildModel2D(num_classes);

    {
        Tensor probe(1, 1, IMAGE_SIZE_2D, IMAGE_SIZE_2D);
        DimensionCalculator::debugArchitecture(model, probe);
    }

    // --- DataLoaders ---
    DataLoader train_loader(train_dataset, BATCH_SIZE_2D, /*shuffle=*/true);
    DataLoader val_loader  (val_dataset,   BATCH_SIZE_2D, /*shuffle=*/false);

    // [TEST RAPIDE] Limiter le nombre d'échantillons
    train_loader.setMaxSamples(100);
    val_loader.setMaxSamples(20);

    // --- Entraînement ---
    section("Entraînement 2D");
    model.fitWithValidation(train_loader, val_loader, EPOCHS_2D, BATCH_SIZE_2D);

    return 0;
}

// =============================================================================
// Pipeline 3D — MedMNIST3D (FractureMNIST3D par défaut)
// =============================================================================

static CNN buildModel3D(int num_classes) {
    CNN model;

    // --- Feature extraction 3D ---
    // Bloc 1 : 1 → 16,  stride 1, pad 1  → (B, 16, 28, 28, 28)
    model.addLayer(std::make_shared<ConvLayer3D>(1,  16, 3, 3, 3, 1, 1, 1, 1, 1, 1));
    model.addLayer(std::make_shared<ReLULayer>());
    // Bloc 2 : 16 → 32, stride 2, pad 1  → (B, 32, 14, 14, 14)
    model.addLayer(std::make_shared<ConvLayer3D>(16, 32, 3, 3, 3, 2, 2, 2, 1, 1, 1));
    model.addLayer(std::make_shared<ReLULayer>());
    // Bloc 3 : 32 → 64, stride 2, pad 1  → (B, 64,  7,  7,  7)
    model.addLayer(std::make_shared<ConvLayer3D>(32, 64, 3, 3, 3, 2, 2, 2, 1, 1, 1));
    model.addLayer(std::make_shared<ReLULayer>());

    // --- Classification ---
    model.addLayer(std::make_shared<DenseLayer>(7 * 7 * 7 * 64, 256));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<DropoutLayer>(0.5f));
    model.addLayer(std::make_shared<DenseLayer>(256, num_classes));
    model.setLossLayer(std::make_shared<SoftmaxCrossEntropyLayer>());

    model.setOptimizer(std::make_shared<Adam>(LR_3D));
    return model;
}

static int run3D() {
    section("Pipeline 3D — FractureMNIST3D");

    requireDir(FRACTURE_PATH);

    // --- Datasets ---
    MedMNIST3DDataset train_dataset(FRACTURE_PATH, Split::TRAIN, 3, "FractureMNIST3D", true);
    MedMNIST3DDataset val_dataset  (FRACTURE_PATH, Split::VAL,   3, "FractureMNIST3D", true);

    std::cout << "Classes    : 3\n"
              << "Train      : " << train_dataset.getNumSamples() << "\n"
              << "Validation : " << val_dataset.getNumSamples()   << "\n";

    if (train_dataset.getNumSamples() == 0)
        throw std::runtime_error("Dataset 3D vide !");

    // --- Modèle ---
    section("Architecture 3D");
    CNN model = buildModel3D(/*num_classes=*/3);

    {
        // Probe 5D : (1, 1, 28, 28, 28)
        Tensor probe(1, 1, VOL_SIZE, VOL_SIZE, VOL_SIZE);
        DimensionCalculator::debugArchitecture(model, probe);
    }

    // --- DataLoaders ---
    DataLoader3D train_loader(train_dataset, BATCH_SIZE_3D, /*shuffle=*/true);
    DataLoader3D val_loader  (val_dataset,   BATCH_SIZE_3D, /*shuffle=*/false);

    // [TEST RAPIDE] Limiter le nombre d'échantillons
    train_loader.setMaxSamples(32);
    val_loader.setMaxSamples(16);

    // --- Entraînement ---
    section("Entraînement 3D");
    model.fitWithValidation(train_loader, val_loader, EPOCHS_3D, BATCH_SIZE_3D);

    return 0;
}

// =============================================================================
// Point d'entrée
// =============================================================================

int main() {
    try {
        Eigen::initParallel();
        Eigen::setNbThreads(4);

        auto t0 = std::chrono::high_resolution_clock::now();

        int ret = 0;
        if constexpr (ACTIVE_PIPELINE == Pipeline::CNN2D) {
            ret = run2D();
        } else {
            ret = run3D();
        }

        auto t1  = std::chrono::high_resolution_clock::now();
        auto min = std::chrono::duration_cast<std::chrono::minutes>(t1 - t0).count();
        auto sec = std::chrono::duration_cast<std::chrono::seconds> (t1 - t0).count() % 60;

        section("Terminé");
        std::cout << "Durée totale : " << min << "m " << sec << "s\n";

        return ret;

    } catch (const std::exception& e) {
        std::cerr << "\n[ERREUR] " << e.what() << "\n";
        return 1;
    }
}