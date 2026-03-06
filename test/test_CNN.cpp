// tests/test_CNN.cpp

#include <gtest/gtest.h>
#include "CNN.hpp"
#include "DenseLayer.hpp"
#include "ActivationLayer.hpp"
#include "LossLayer.hpp"
#include "Optimizer.hpp"

// =============================================================================
// ── Construction et Structure ────────────────────────────────────────────────
// =============================================================================

TEST(CNNBuild, AddLayer_IncreasesCount) {
    CNN model;
    EXPECT_EQ(model.getLayers().size(), 0);
    model.addLayer(std::make_shared<DenseLayer>(10, 5));
    EXPECT_EQ(model.getLayers().size(), 1);
    model.addLayer(std::make_shared<ReLULayer>());
    EXPECT_EQ(model.getLayers().size(), 2);
}

TEST(CNNBuild, GetLayer_ReturnsCorrectLayer) {
    CNN model;
    model.addLayer(std::make_shared<DenseLayer>(10, 5));
    model.addLayer(std::make_shared<ReLULayer>());
    
    EXPECT_EQ(model.getLayer(0)->getName(), "Dense");
    EXPECT_EQ(model.getLayer(1)->getName(), "ReLU");
}

TEST(CNNBuild, GetLayer_OutOfBoundsThrows) {
    CNN model;
    model.addLayer(std::make_shared<DenseLayer>(10, 5));
    EXPECT_THROW(model.getLayer(1), std::out_of_range);
    EXPECT_THROW(model.getLayer(-1), std::out_of_range);
}

TEST(CNNBuild, SetLossLayer_Success) {
    CNN model;
    auto loss = std::make_shared<CrossEntropyLoss>();
    model.setLossLayer(loss);
    EXPECT_EQ(model.getLossLayer()->getName(), "CrossEntropyLoss");
}

// =============================================================================
// ── Forward et Backward (Architecture Minimale) ──────────────────────────────
// =============================================================================

class CNNTestFixture : public ::testing::Test {
protected:
    CNN model;
    
    void SetUp() override {
        // Architecture: Input (B, 4) -> Dense(4, 2) -> SoftmaxCE
        model.addLayer(std::make_shared<DenseLayer>(4, 2));
        model.setLossLayer(std::make_shared<SoftmaxCrossEntropyLayer>());
        model.setOptimizer(std::make_shared<SGD>(0.01f));
    }
};

TEST_F(CNNTestFixture, ForwardShape) {
    Tensor input(1, 4, 1, 1);
    input.setRandom();
    
    Tensor out = model.forward(input);
    EXPECT_EQ(out.dim(0), 1);
    EXPECT_EQ(out.dim(1), 2);
    EXPECT_EQ(out.dim(2), 1);
    EXPECT_EQ(out.dim(3), 1);
}

TEST_F(CNNTestFixture, BackwardRunsWithoutError) {
    Tensor input(1, 4, 1, 1);
    input.setRandom();
    
    // Simuler des cibles pour le backward
    Tensor target(1, 2, 1, 1);
    target.setZero();
    target(0, 0, 0, 0) = 1.0f; // Vraie classe = 0
    
    model.getLossLayer()->setTargets(target);
    model.forward(input);
    
    // Le backward retourne un tenseur (gradient par rapport à l'entrée)
    Tensor grad_in = model.backward(Tensor()); // Dummy pour déclencher le pipeline
    
    EXPECT_EQ(grad_in.size(), input.size());
}

TEST_F(CNNTestFixture, UpdateWeights_AltersParameters) {
    // Test d'un pas d'apprentissage
    auto dense = std::dynamic_pointer_cast<DenseLayer>(model.getLayers()[0]);
    Eigen::MatrixXf W_before = dense->getWeights();
    
    Tensor input(1, 4, 1, 1);
    input.setRandom();
    Tensor target(1, 2, 1, 1);
    target.setZero();
    target(0, 1, 0, 0) = 1.0f; 
    
    model.getLossLayer()->setTargets(target);
    model.forward(input);
    model.backward(Tensor());
    model.updateWeights();
    
    EXPECT_GT((dense->getWeights() - W_before).norm(), 0.0f);
}

TEST_F(CNNTestFixture, PredictReturnsTensor) {
    Tensor input(1, 4, 1, 1);
    input.setRandom();
    Tensor preds = model.predict(input);
    
    EXPECT_EQ(preds.dim(0), 1);  // Batch size
    EXPECT_EQ(preds.dim(1), 1);  // Un seul entier (indice de classe)
}

TEST_F(CNNTestFixture, Evaluate_ReturnsAccuracy) {
    Tensor input(1, 4, 1, 1);
    input.setRandom();
    Tensor target(1, 2, 1, 1);
    target.setZero();
    target(0, 0, 0, 0) = 1.0f;
    
    // N'entraîne pas de crash et retourne [0, 1]
    float acc = model.evaluate(input, target);
    EXPECT_GE(acc, 0.0f);
    EXPECT_LE(acc, 1.0f);
}
