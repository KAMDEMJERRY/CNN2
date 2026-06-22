#pragma once

#include "CNNLIB.hpp"
#include <vector>
#include <memory>
#include <chrono>

#include "PatchifyStem3D.hpp"
#include "ConvNeXtBlock3D.hpp"
#include "DenseDownsample3D.hpp"
#include "PatchifyStem3DSparse.hpp"
#include "ConvNeXtBlock3DSparse.hpp"
#include "SparseDownsample3D.hpp"

// =============================================================================
// Configuration globale
// =============================================================================

// --- 2D ---
// #define MNIST_TEST_PATH    "../dataset/mnist_img/trainingSet/trainingSet/" 
// #define MNIST_TRAIN_PATH    "../dataset/mnist_img/trainingSample/trainingSample/"
// #define BLOODCELLS_TRAIN_PATH "../dataset/bloodcell/images/TRAIN/"
// #define BLOODCELLS_TEST_PATH  "../dataset/bloodcell/images/TEST/"

static constexpr bool  USE_BLOODCELLS = false;
static constexpr int   IMAGE_SIZE_2D = 28;
static constexpr int   BATCH_SIZE_2D = 10;
static constexpr int   EPOCHS_2D = 50;
static constexpr float LR_2D = 0.0001f;

// --- 3D dense ---
// #define FRACTURE_PATH "../dataset/fracturemnist3d" 
// #define NODULE_PATH "../dataset/nodulemnist3d"
// #define VESSEL_PATH "../dataset/vesselmnist3d"
// #define ADRENAL_PATH "../dataset/adrenalmnist3d"

static constexpr int   BATCH_SIZE_3D = 16;
static constexpr int   EPOCHS_3D = 150;
static constexpr float LR_3D = 3e-4f;
static constexpr int   VOL_SIZE = 64;

// --- 3D sparse ---
static constexpr float SPARSE_THRESHOLD = 0.02f;
static constexpr int   BATCH_SIZE_SPARSE = 16;
static constexpr int   EPOCHS_SPARSE = 150;
static constexpr float LR_SPARSE = 0.0003f;

// --- Attention ---
static constexpr int   ATTN_WIN_LARGE = 7;
static constexpr int   ATTN_WIN_SMALL = 4;
static constexpr int   ATTN_HEADS = 4;

// -- Scheduler ---
static constexpr bool schedule = true;

inline CNN buildModel2D(int num_classes) {
    CNN model;
    model.addLayer(std::make_shared<ConvLayerDataParallel>(1, 32, 3, 3, 1, 1, 1, 1));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<ConvLayerDataParallel>(32, 32, 3, 3, 1, 1, 1, 1));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<MaxPoolLayer>(2, 2));
    model.addLayer(std::make_shared<ConvLayerDataParallel>(32, 64, 3, 3, 1, 1, 1, 1));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<ConvLayerDataParallel>(64, 64, 3, 3, 1, 1, 1, 1));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<MaxPoolLayer>(2, 2));

    int n = IMAGE_SIZE_2D / 4;
    model.addLayer(std::make_shared<DenseLayer>(n * n * 64, 128));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<DropoutLayer>(0.5f));
    model.addLayer(std::make_shared<DenseLayer>(128, num_classes));
    model.setLossLayer(std::make_shared<SoftmaxCrossEntropyLayer>());
    model.setOptimizer(std::make_shared<Adam>(LR_2D));
    if(schedule){
    model.setScheduler(std::make_shared<CosineDecay>());
    std::cout << "\n===> shcheduler set" << std::endl;
    }
    return model;
}

inline CNN buildModel3D(int num_classes) {
    CNN model;
    model.addLayer(std::make_shared<ConvLayer3D>(1, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1));
    model.addLayer(std::make_shared<BatchNorm3D>(16));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<ConvLayer3D>(16, 32, 3, 3, 3, 2, 2, 2, 1, 1, 1));
    model.addLayer(std::make_shared<BatchNorm3D>(32));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<ConvLayer3D>(32, 64, 3, 3, 3, 2, 2, 2, 1, 1, 1));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<DenseLayer>(7 * 7 * 7 * 64, 256));
    model.addLayer(std::make_shared<BatchNorm3D>(256));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<DropoutLayer>(0.5f));
    model.addLayer(std::make_shared<DenseLayer>(256, num_classes));
    
    std::vector<float> weights = {0.723f, 0.894f, 2.002f}; 
    int gamma = 2;
    model.setLossLayer(std::make_shared<SoftmaxCrossEntropyLayer>(weights, gamma));
    
    model.setOptimizer(std::make_shared<Adam>(LR_3D));
    if(schedule){
    model.setScheduler(std::make_shared<CosineDecay>());
    std::cout << "\n===> shcheduler set" << std::endl;
    }

    return model;
}

inline CNN buildModel3DSparse(int num_classes) {
    CNN model;
    model.addLayer(std::make_shared<SparseConvAdapterLayer>(
        1, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1, true, SPARSE_THRESHOLD));
    model.addLayer(std::make_shared<BatchNorm3D>(16));
    model.addLayer(std::make_shared<ReLULayer>());

    model.addLayer(std::make_shared<SparseConvAdapterLayer>(
        16, 32, 3, 3, 3, 2, 2, 2, 1, 1, 1, false, 0.0f));
    model.addLayer(std::make_shared<BatchNorm3D>(32));
    model.addLayer(std::make_shared<ReLULayer>());

    model.addLayer(std::make_shared<SparseConvAdapterLayer>(
        32, 64, 3, 3, 3, 2, 2, 2, 1, 1, 1, false, 0.0f));
    model.addLayer(std::make_shared<BatchNorm3D>(64));
    model.addLayer(std::make_shared<ReLULayer>());

    model.addLayer(std::make_shared<SparseGlobalAvgPoolLayer>(0.0f));
    model.addLayer(std::make_shared<DenseLayer>(64, 256));
    model.addLayer(std::make_shared<ReLULayer>());            
    model.addLayer(std::make_shared<DropoutLayer>(0.5f));
    model.addLayer(std::make_shared<DenseLayer>(256, num_classes));

    std::vector<float> weights ={0.723f, 0.894f, 2.002f};  
    float gamma = 2.0f;
    model.setLossLayer(std::make_shared<SoftmaxCrossEntropyLayer>(weights, gamma));

    model.setOptimizer(std::make_shared<Adam>(LR_SPARSE));
    if(schedule){
        model.setScheduler(std::make_shared<CosineDecay>());
        std::cout << "\n===> shcheduler set" << std::endl;
    }

    return model;
}

inline CNN buildModel3DAttn(int num_classes) {
    CNN model;
    model.addLayer(std::make_shared<ConvLayer3D>(1, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1));
    model.addLayer(std::make_shared<BatchNorm3D>(16));
    model.addLayer(std::make_shared<ReLULayer>());

    model.addLayer(std::make_shared<ConvLayer3D>(16, 32, 3, 3, 3, 2, 2, 2, 1, 1, 1));
    model.addLayer(std::make_shared<BatchNorm3D>(32));
    model.addLayer(std::make_shared<ReLULayer>());

    model.addLayer(std::make_shared<WindowAttention3DLayer>(
        32, ATTN_WIN_SMALL, ATTN_WIN_SMALL, ATTN_WIN_SMALL,
        ATTN_HEADS, true, true));

    model.addLayer(std::make_shared<ConvLayer3D>(32, 64, 3, 3, 3, 2, 2, 2, 1, 1, 1));
    model.addLayer(std::make_shared<BatchNorm3D>(64));
    model.addLayer(std::make_shared<ReLULayer>());

    model.addLayer(std::make_shared<WindowAttention3DLayer>(
        64, ATTN_WIN_LARGE, ATTN_WIN_LARGE, ATTN_WIN_LARGE,
        ATTN_HEADS, true, true));

    model.addLayer(std::make_shared<GlobalAvgPool3DLayer>());
    model.addLayer(std::make_shared<DenseLayer>(64, 256));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<DropoutLayer>(0.3f));
    model.addLayer(std::make_shared<DenseLayer>(256, num_classes));

    std::vector<float> weights = {0.723f, 0.894f, 2.002f}; 
    float gamma = 2.0f;
    model.setLossLayer(std::make_shared<SoftmaxCrossEntropyLayer>(weights, gamma));

    model.setOptimizer(std::make_shared<Adam>(LR_3D));
    if(schedule){
        model.setScheduler(std::make_shared<CosineDecay>());
        std::cout << "\n===> shcheduler set" << std::endl;
    }

    return model;
}

inline CNN buildModel3DSparseAttn(int num_classes) {
    CNN model;
    model.addLayer(std::make_shared<SparseConvAdapterLayer>(
        1, 16, 3,3,3, 1,1,1, 1,1,1, true, SPARSE_THRESHOLD));
    model.addLayer(std::make_shared<BatchNorm3D>(16));
    model.addLayer(std::make_shared<SparseReLULayer>(0.0f));

    model.addLayer(std::make_shared<SparseConvAdapterLayer>(
        16, 32, 3,3,3, 2,2,2, 1,1,1, false, 0.0f));
    model.addLayer(std::make_shared<BatchNorm3D>(32));
    model.addLayer(std::make_shared<SparseReLULayer>(0.0f));

    model.addLayer(std::make_shared<WindowAttention3DLayer>(
        32, ATTN_WIN_SMALL, ATTN_WIN_SMALL, ATTN_WIN_SMALL,
        ATTN_HEADS, true, true));

    model.addLayer(std::make_shared<SparseConvAdapterLayer>(
        32, 64, 3,3,3, 2,2,2, 1,1,1, false, 0.0f));
    model.addLayer(std::make_shared<BatchNorm3D>(64));
    model.addLayer(std::make_shared<SparseReLULayer>(0.0f));

    model.addLayer(std::make_shared<WindowAttention3DLayer>(
        64, ATTN_WIN_LARGE, ATTN_WIN_LARGE, ATTN_WIN_LARGE,
        ATTN_HEADS, true, true));

    model.addLayer(std::make_shared<GlobalAvgPool3DLayer>());
    model.addLayer(std::make_shared<DenseLayer>(64, 256));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<DropoutLayer>(0.3f));
    model.addLayer(std::make_shared<DenseLayer>(256, num_classes));

    const std::vector<float> weights = {0.723f, 0.894f, 2.002f};
    float gamma = 2.0f;
    model.setLossLayer(
        std::make_shared<SoftmaxCrossEntropyLayer>(weights, gamma));

    model.setOptimizer(std::make_shared<Adam>(LR_SPARSE));
    if(schedule){
        model.setScheduler(std::make_shared<CosineDecay>());
        std::cout << "\n===> shcheduler set" << std::endl;
    }

    return model;
}


// inline CNN buildConvNeXt3DDense(int num_classes) {
//     CNN model;
   
//     // Stem : 28³ → 14³, 1 canal → 32
//     model.addLayer(std::make_shared<PatchifyStem3D>(1, 32, 2));

//     // Stage 1 : 14³, kernel 3
//     model.addLayer(std::make_shared<ConvNeXtBlock3D>(32, 3));
//     model.addLayer(std::make_shared<ConvNeXtBlock3D>(32, 3));

//     // Downsample : 14³ → 7³, 32 → 64
//     model.addLayer(std::make_shared<DenseDownsample3D>(32, 64));

//     // Stage 2 : 7³, kernel 3
//     model.addLayer(std::make_shared<ConvNeXtBlock3D>(64, 3));
//     model.addLayer(std::make_shared<ConvNeXtBlock3D>(64, 3));

//     // Tête de classification
//     model.addLayer(std::make_shared<GlobalAvgPool3DLayer>());
//     model.addLayer(std::make_shared<DenseLayer>(64, num_classes));
    
//     std::vector<float> weights = {0.723f, 0.894f, 2.002f};
//     int gamma = 2;
//     model.setLossLayer(std::make_shared<SoftmaxCrossEntropyLayer>(weights, gamma));
    
//     model.setOptimizer(std::make_shared<Adam>(LR_3D));
//     if(schedule){
//         model.setScheduler(std::make_shared<CosineDecay>());
//         std::cout << "\n===> shcheduler set" << std::endl;
//     }

//     return model;
// }

inline CNN buildConvNeXt3DDense(int num_classes) {
    CNN model;

    // ── Stem : 64³ → 32³, 1 → 48ch, stride 2
    model.addLayer(std::make_shared<PatchifyStem3D>(1, 48, 2));

    // ── Stage 1 : 32³, kernel 7 (champ réceptif large dès le début)
    model.addLayer(std::make_shared<ConvNeXtBlock3D>(48, 7));
    model.addLayer(std::make_shared<ConvNeXtBlock3D>(48, 7));
    model.addLayer(std::make_shared<ConvNeXtBlock3D>(48, 7));

    // ── Downsample 1 : 32³ → 16³, 48 → 96ch
    model.addLayer(std::make_shared<DenseDownsample3D>(48, 96));

    // ── Stage 2 : 16³, kernel 7
    model.addLayer(std::make_shared<ConvNeXtBlock3D>(96, 7));
    model.addLayer(std::make_shared<ConvNeXtBlock3D>(96, 7));
    model.addLayer(std::make_shared<ConvNeXtBlock3D>(96, 7));

    // ── Downsample 2 : 16³ → 8³, 96 → 192ch
    model.addLayer(std::make_shared<DenseDownsample3D>(96, 192));

    // ── Stage 3 : 8³, kernel 7
    model.addLayer(std::make_shared<ConvNeXtBlock3D>(192, 7));
    model.addLayer(std::make_shared<ConvNeXtBlock3D>(192, 7));
    model.addLayer(std::make_shared<ConvNeXtBlock3D>(192, 7));

    // ── Tête : LayerNorm → GlobalAvgPool → Classifier
    model.addLayer(std::make_shared<LayerNorm3DLayer>(192));
    model.addLayer(std::make_shared<GlobalAvgPool3DLayer>());
    model.addLayer(std::make_shared<DenseLayer>(192, num_classes));

    // ── Loss + Adam + Cosine decay (identique à ton code)
    std::vector<float> weights = {0.723f, 0.894f, 2.002f};
    int gamma = 2;
    model.setLossLayer(
        std::make_shared<SoftmaxCrossEntropyLayer>(weights, gamma));
    model.setOptimizer(std::make_shared<Adam>(LR_3D));
    if (schedule) {
        model.setScheduler(std::make_shared<CosineDecay>());
        std::cout << "\n===> scheduler set" << std::endl;
    }
    return model;
}


inline CNN buildConvNeXt3DSparse(int num_classes) {
    CNN model;
    model.addLayer(std::make_shared<PatchifyStem3DSparse>(1, 32, 2, SPARSE_THRESHOLD));
    model.addLayer(std::make_shared<ConvNeXtBlock3DSparse>(32, 7, 1e-6f, 0.0f));
    model.addLayer(std::make_shared<SparseDownsample3D>(32, 64, 0.0f));
    model.addLayer(std::make_shared<ConvNeXtBlock3DSparse>(64, 7, 1e-6f, 0.0f));
    model.addLayer(std::make_shared<GlobalAvgPool3DLayer>());
    model.addLayer(std::make_shared<DenseLayer>(64, num_classes));
    
    std::vector<float> weights = {0.723f, 0.894f, 2.002f};
    int gamma = 2; // Focal loss
    model.setLossLayer(std::make_shared<SoftmaxCrossEntropyLayer>(weights, gamma));
    
    model.setOptimizer(std::make_shared<Adam>(LR_SPARSE));
    if(schedule){
      model.setScheduler(std::make_shared<CosineDecay>());
      std::cout << "\n===> shcheduler set" << std::endl;
    }
    return model;
}
