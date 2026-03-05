// main.cpp
#include "ConvLayer.hpp"
#include "ActivationLayer.hpp"
#include "MaxPoolLayer.hpp"
#include "DenseLayer.hpp"
#include "LossLayer.hpp"
#include "Optimizer.hpp"
#include "Dimensions.hpp"
#include "DataLoader.hpp"
#include "DropoutLayer.hpp"
#include "CNN.hpp"
#include <iostream>
#include <chrono>
#include <filesystem>

// Dataset paths
#define MNIST_TRAIN_PATH   "../../../dataset/mnist_img/trainingSample/trainingSample/"
#define MNIST_TEST_PATH    "../../../dataset/mnist_img/trainingSample/trainingSample/"
#define BLOODCELLS_TRAIN_PATH  "../../../dataset/bloodcell/images/TRAIN/"
#define BLOODCELLS_TEST_PATH   "../../../dataset/bloodcell/images/TEST/"

// Configuration
#define USE_BLOODCELLS false  // Set false for MNIST
#define IMAGE_SIZE 28
#define BATCH_SIZE 100
#define EPOCHS 50
#define LEARNING_RATE 0.0001f

int main() {
    try {
        // 1. Initialisation
        Eigen::initParallel();
        Eigen::setNbThreads(4);
        
        std::cout << "=== CNN Training Pipeline ===" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        // 2. Charger le dataset
        std::string train_data_dir = USE_BLOODCELLS ? BLOODCELLS_TRAIN_PATH : MNIST_TRAIN_PATH;
        std::string test_data_dir = USE_BLOODCELLS ? BLOODCELLS_TEST_PATH : MNIST_TEST_PATH;
        
        std::cout << "Dataset: " << (USE_BLOODCELLS ? "Blood Cells" : "MNIST") << std::endl;
        std::cout << "Train path: " << train_data_dir << std::endl;
        std::cout << "Test path: " << test_data_dir << std::endl;
        
        // Check if directories exist
        if (!std::filesystem::exists(train_data_dir)) {
            throw std::runtime_error("Train directory does not exist: " + train_data_dir);
        }
        if (!std::filesystem::exists(test_data_dir)) {
            throw std::runtime_error("Test directory does not exist: " + test_data_dir);
        }

        // Load datasets
        ImageFolderDataset train_dataset(train_data_dir, IMAGE_SIZE, IMAGE_SIZE, true, true);
        ImageFolderDataset val_dataset(test_data_dir, IMAGE_SIZE, IMAGE_SIZE, false, true);

        int num_classes = train_dataset.getNumClasses();
        int num_train_samples = train_dataset.getNumSamples();
        int num_val_samples = val_dataset.getNumSamples();
        
        std::cout << "Classes: " << num_classes << std::endl;
        std::cout << "Training images: " << num_train_samples << std::endl;
        std::cout << "Validation images: " << num_val_samples << std::endl;

        if (num_train_samples == 0 || num_val_samples == 0) {
            throw std::runtime_error("Dataset is empty!");
        }

        // 3. Créer le modèle
        std::cout << "\nBuilding model architecture..." << std::endl;
        CNN model;

        // Feature extraction
        model.addLayer(std::make_unique<ConvLayer>(1, 32, 3, 3, 1, 1, 1, 1));
        model.addLayer(std::make_unique<ReLULayer>());
        model.addLayer(std::make_unique<ConvLayer>(32, 32, 3, 3, 1, 1, 1, 1));
        model.addLayer(std::make_unique<ReLULayer>());
        model.addLayer(std::make_unique<MaxPoolLayer>(2, 2));  // 14x14

        model.addLayer(std::make_unique<ConvLayer>(32, 64, 3, 3, 1, 1, 1, 1));
        model.addLayer(std::make_unique<ReLULayer>());
        model.addLayer(std::make_unique<ConvLayer>(64, 64, 3, 3, 1, 1, 1, 1));
        model.addLayer(std::make_unique<ReLULayer>());
        model.addLayer(std::make_unique<MaxPoolLayer>(2, 2));  // 7x7

        // Classification
        model.addLayer(std::make_unique<DenseLayer>(7 * 7 * 64, 128));
        model.addLayer(std::make_unique<ReLULayer>());
        model.addLayer(std::make_unique<DropoutLayer>(0.5f));
        model.addLayer(std::make_unique<DenseLayer>(128, num_classes));

        // Fonction de perte et optimiseur
        auto softmax_ce = std::make_unique<SoftmaxCrossEntropyLayer>();
        model.addLayer(std::move(softmax_ce));

        model.setOptimizer(std::make_shared<Adam>(LEARNING_RATE));

        // Optional: Debug architecture with a dummy tensor
        if constexpr (true) {  // Set to true to enable debugging
            std::cout << "\nDebugging architecture..." << std::endl;
            Tensor test_input(1, 1, IMAGE_SIZE, IMAGE_SIZE);
            DimensionCalculator::debugArchitecture(model, test_input);
        }

        // 4. Entraînement
        std::cout << "\nStarting training..." << std::endl;
        DataLoader train_loader(train_dataset, BATCH_SIZE, true);
        DataLoader val_loader(val_dataset, BATCH_SIZE, false);

        // Train with validation
        // model.fitWithValidation(train_loader, val_loader, EPOCHS, 100);
        model.fit(train_loader, EPOCHS, 100);
        
        // Optional: Final evaluation
        // std::cout << "\nFinal evaluation on validation set:" << std::endl;
        // auto metrics = model.evaluate(val_loader);
        
        // auto end_time = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
        
        // std::cout << "\n=== Training Complete ===" << std::endl;
        // std::cout << "Total time: " << duration.count() << " minutes" << std::endl;
        // std::cout << "Final validation accuracy: " << (metrics.accuracy * 100) << "%" << std::endl;
        // std::cout << "Final validation loss: " << metrics.loss << std::endl;

        // Optional: Save model
        // model.save("trained_model.bin");

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}