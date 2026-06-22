// main.cpp
#include "shared.hpp"
#include "CNNLIB.hpp"
#include "ModelBuilders.hpp"
#include "DatasetManager.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <fstream>

#define PATIENCE 30
#define AUGMENT_DATA true
#define MAX_SAMPLE false
#define MAX_SAMPLE_TRAIN 32
#define MAX_SAMPLE_VAL 32

enum class Pipeline { CNN2D, CNN3D, CNN3D_SPARSE, CNN3D_ATTN, CNN3D_SPARSE_ATTN, CNN3D_CONVNEXT_DENSE, CNN3D_CONVNEXT_SPARSE };
static Pipeline ACTIVE_PIPELINE = Pipeline::CNN2D;
static std::chrono::high_resolution_clock::time_point START_TIME;

// =============================================================================
// Configuration centralisée des datasets
// =============================================================================

// Pour les pipelines 3D - sélectionnez votre dataset ici
// const DatasetManager::DatasetType3D ACTIVE_DATASET_3D = DatasetManager::FRACTURE_3D;
const DatasetManager::DatasetType3D ACTIVE_DATASET_3D = DatasetManager::FRACTURE_3D_64;
// const DatasetManager::DatasetType3D ACTIVE_DATASET_3D = DatasetManager::NODULE_3D;
// const DatasetManager::DatasetType3D ACTIVE_DATASET_3D = DatasetManager::VESSEL_3D;
// const DatasetManager::DatasetType3D ACTIVE_DATASET_3D = DatasetManager::ADRENAL_3D;
// const DatasetManager::DatasetType3D ACTIVE_DATASET_3D = DatasetManager::ADRENAL_3D_64;
// const DatasetManager::DatasetType3D ACTIVE_DATASET_3D = DatasetManager::MEDMNIST_3D;

// Pour les pipelines 2D - sélectionnez votre dataset ici
const DatasetManager::DatasetType2D ACTIVE_DATASET_2D = DatasetManager::MNIST_2D;
// const DatasetManager::DatasetType2D ACTIVE_DATASET_2D = DatasetManager::BLOODCELLS_2D;

// =============================================================================
// Fonctions utilitaires
// =============================================================================

static void printDatasetInfo(const DatasetManager::Info& dataset) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Dataset: " << dataset.name << std::endl;
    std::cout << "Path: " << (dataset.path.empty() ? "Separate train/test paths" : dataset.path) << std::endl;
    std::cout << "Dimensions: " << dataset.dims << "D" << std::endl;
    std::cout << "Classes: " << dataset.num_classes << std::endl;
    std::cout << "Channels: " << dataset.num_channels << std::endl;
    
    if (dataset.dims == 2) {
        std::cout << "Image size: " << dataset.img_height << "x" << dataset.img_width << std::endl;
    } else if (dataset.dims == 3) {
        std::cout << "Volume size: " << dataset.vol_size << "x" << dataset.vol_size << "x" << dataset.vol_size << std::endl;
    }
    
    if (!dataset.class_names.empty()) {
        std::cout << "Class names: ";
        for (size_t i = 0; i < dataset.class_names.size(); ++i) {
            std::cout << dataset.class_names[i];
            if (i < dataset.class_names.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "========================================\n" << std::endl;
}

// =============================================================================
// Pipeline 2D - Complètement dynamique
// =============================================================================
static int run2D(bool trainBeforeTesting = true) {
    // Récupérer la configuration du dataset 2D actif
    auto dataset = DatasetManager::getInfo(ACTIVE_DATASET_2D);
    auto paths = DatasetManager::getPaths2D(ACTIVE_DATASET_2D);
    
    section("Pipeline 2D - " + dataset.name);
    
    const std::string train_dir = paths.first;
    const std::string test_dir = paths.second;
    
    requireDir(train_dir);
    requireDir(test_dir);
    
    printDatasetInfo(dataset);
    
    // Création des datasets avec les dimensions appropriées
    ImageFolderDataset train_ds(train_dir, dataset.img_height, dataset.img_width, true, true);
    ImageFolderDataset val_ds(test_dir, dataset.img_height, dataset.img_width, false, true);
    
    // Vérification que le nombre de classes correspond
    int ds_num_classes = train_ds.getNumClasses();
    if (ds_num_classes != dataset.num_classes) {
        std::cout << "Warning: Dataset reports " << ds_num_classes 
                  << " classes but config expects " << dataset.num_classes << std::endl;
    }
    
    // Construction du modèle avec le bon nombre de classes
    CNN model = buildModel2D(dataset.num_classes);
    
    DataLoader train_loader(train_ds, BATCH_SIZE_2D, true);
    DataLoader val_loader(val_ds, BATCH_SIZE_2D, false);

    // if(AUGMENT_DATA){
    //     train_loader.setAugmentation();
    //     std::cout << "\n===> Data Augmented" << std::endl;
    // }

    if(MAX_SAMPLE){
        train_loader.setMaxSamples(MAX_SAMPLE_TRAIN); 
        val_loader.setMaxSamples(MAX_SAMPLE_VAL);
        std::cout << "\n===> Max samples : (" << MAX_SAMPLE_TRAIN << ") :Train (" << MAX_SAMPLE_VAL << ") :VAL " << std::endl;
    }
    
    std::string filename = "./models/2d_best_" + dataset.name + ".bin";
    
    if (trainBeforeTesting) {
        EarlyStopping es;
        es.patience = PATIENCE;
        es.min_delta = 5e-4f;
        es.checkpoint = filename;
        es.log_file = "./logs/train_2d_" + dataset.name + ".txt";
        model.fitWithValidation(train_loader, val_loader, EPOCHS_2D, BATCH_SIZE_2D, es);
        model.saveParameters(filename);
    } else {
        if(fs::exists(filename)) model.loadParameters(filename);
    }

    section("Évaluation finale sur le Test Set");
    std::string log_filename = "./logs/eval_2d_" + dataset.name + ".txt";
    std::cout << "Saving evaluation results to " << log_filename << "\n";
    requireDir("./logs");
    std::ofstream log_file(log_filename);
    TeeBuffer tee(std::cout.rdbuf(), log_file.rdbuf(), true);
    std::ostream tee_stream(&tee);
    std::streambuf* coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(tee_stream.rdbuf());
    
    model.evaluate(val_loader);

    val_loader.reset();
    auto cm = model.confusionMatrix(val_loader, dataset.num_classes);
    model.printConfusionMatrix(cm, dataset.class_names);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto min = std::chrono::duration_cast<std::chrono::minutes>(t1 - START_TIME).count();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(t1 - START_TIME).count() % 60;
    std::cout << "\nDurée totale : " << min << "m " << sec << "s\n";

    std::cout.rdbuf(coutbuf);
    log_file.close();

    val_loader.reset();
    model.exportPredictionsToCSV(val_loader, "./logs/predictions_2d_" + dataset.name + ".csv");

    return 0;
}

// =============================================================================
// Pipeline 3D dense - Refactorisé
// =============================================================================
static int run3D(bool trainBeforeTesting = true) {
    auto dataset = DatasetManager::getInfo(ACTIVE_DATASET_3D);
    
    section("Pipeline 3D dense - " + dataset.name);
    requireDir(dataset.path);
    
    std::string filename = "./models/3d_dense_" + dataset.name + ".bin";
    printDatasetInfo(dataset);
    
    MedMNIST3DDataset train_ds(dataset.path, Split::TRAIN, dataset.num_classes, dataset.name, true);
    MedMNIST3DDataset val_ds(dataset.path, Split::VAL, dataset.num_classes, dataset.name, true);
    MedMNIST3DDataset test_ds(dataset.path, Split::TEST, dataset.num_classes, dataset.name, true);
    
    CNN model = buildModel3D(dataset.num_classes);
    
    {
        Tensor probe(1, 1, VOL_SIZE, VOL_SIZE, VOL_SIZE);
        DimensionCalculator::debugArchitecture(model, probe);
    }

    DataLoader3D train_loader(train_ds, BATCH_SIZE_3D, true);
    DataLoader3D val_loader(val_ds, BATCH_SIZE_3D, false);
    DataLoader3D test_loader(test_ds, BATCH_SIZE_3D, false);

    if(AUGMENT_DATA){
        train_loader.setAugmentation();
        std::cout << "\n===> Data Augmented" << std::endl;
    }

    if(MAX_SAMPLE){
        train_loader.setMaxSamples(MAX_SAMPLE_TRAIN); 
        val_loader.setMaxSamples(MAX_SAMPLE_VAL);
        std::cout << "\n===> Max samples : (" << MAX_SAMPLE_TRAIN << ") :Train (" << MAX_SAMPLE_VAL << ") :VAL " << std::endl;
    }
    
    if (trainBeforeTesting) {
        EarlyStopping es;
        es.patience = PATIENCE;
        es.min_delta = 5e-4f;
        es.checkpoint = filename;
        es.log_file = "./logs/train_3d_dense_" + dataset.name + ".txt";
        model.fitWithValidation(train_loader, val_loader, EPOCHS_3D, BATCH_SIZE_3D, es);
        model.saveParameters(filename);
    } else {
        if(fs::exists(filename)) model.loadParameters(filename);
    }
    
    section("Évaluation finale sur le Test Set");
    std::string log_filename = "./logs/eval_3d_dense_" + dataset.name + ".txt";
    std::cout << "Saving evaluation results to " << log_filename << "\n";
    requireDir("./logs");
    std::ofstream log_file(log_filename);
    TeeBuffer tee(std::cout.rdbuf(), log_file.rdbuf(), true);
    std::ostream tee_stream(&tee);
    std::streambuf* coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(tee_stream.rdbuf());

    model.evaluate(test_loader);

    test_loader.reset();
    auto cm = model.confusionMatrix(test_loader, dataset.num_classes);
    model.printConfusionMatrix(cm, dataset.class_names);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto min = std::chrono::duration_cast<std::chrono::minutes>(t1 - START_TIME).count();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(t1 - START_TIME).count() % 60;
    std::cout << "\nDurée totale : " << min << "m " << sec << "s\n";

    std::cout.rdbuf(coutbuf);
    log_file.close();

    test_loader.reset();
    model.exportPredictionsToCSV(test_loader, "./logs/predictions_3d_dense_" + dataset.name + ".csv");

    return 0;
}

// =============================================================================
// Pipeline 3D sparse - Refactorisé
// =============================================================================
static int run3DSparse(bool trainBeforeTesting = true) {
    auto dataset = DatasetManager::getInfo(ACTIVE_DATASET_3D);

    section("Pipeline 3D sparse (sans attention) - " + dataset.name);
    requireDir(dataset.path);

    std::string filename = "./models/3d_sparse_" + dataset.name + ".bin";
    printDatasetInfo(dataset);

    MedMNIST3DDataset train_ds(dataset.path, Split::TRAIN, dataset.num_classes, dataset.name, true);
    MedMNIST3DDataset val_ds(dataset.path, Split::VAL, dataset.num_classes, dataset.name, true);
    MedMNIST3DDataset test_ds(dataset.path, Split::TEST, dataset.num_classes, dataset.name, true);

    CNN model = buildModel3DSparse(dataset.num_classes);

    {
        Tensor probe(1, 1, VOL_SIZE, VOL_SIZE, VOL_SIZE);
        DimensionCalculator::debugArchitecture(model, probe);
    }

    DataLoader3D train_loader(train_ds, BATCH_SIZE_SPARSE, true);
    DataLoader3D val_loader(val_ds, BATCH_SIZE_SPARSE, false);
    DataLoader3D test_loader(test_ds, BATCH_SIZE_SPARSE, false);

    if(AUGMENT_DATA){
        train_loader.setAugmentation();
        std::cout << "\n===> Data Augmented" << std::endl;
    }

    if(MAX_SAMPLE){
        train_loader.setMaxSamples(MAX_SAMPLE_TRAIN); 
        val_loader.setMaxSamples(MAX_SAMPLE_VAL);
        std::cout << "\n===> Max samples : (" << MAX_SAMPLE_TRAIN << ") :Train (" << MAX_SAMPLE_VAL << ") :VAL " << std::endl;
    }

    if (trainBeforeTesting) {
        EarlyStopping es;
        es.patience = PATIENCE;
        es.min_delta = 5e-4f;
        es.checkpoint = filename;
        es.log_file = "./logs/train_3d_sparse_" + dataset.name + ".txt";
        model.fitWithValidation(train_loader, val_loader, EPOCHS_SPARSE, BATCH_SIZE_SPARSE, es);
        model.saveParameters(filename);
    } else {
        if(fs::exists(filename)) model.loadParameters(filename);
    }
    
    section("Évaluation finale sur le Test Set");
    std::string log_filename = "./logs/eval_3d_sparse_" + dataset.name + ".txt";
    std::cout << "Saving evaluation results to " << log_filename << "\n";
    requireDir("./logs");
    std::ofstream log_file(log_filename);
    TeeBuffer tee(std::cout.rdbuf(), log_file.rdbuf(), true);
    std::ostream tee_stream(&tee);
    std::streambuf* coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(tee_stream.rdbuf());

    model.evaluate(test_loader);

    test_loader.reset();
    auto cm = model.confusionMatrix(test_loader, dataset.num_classes);
    model.printConfusionMatrix(cm, dataset.class_names);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto min = std::chrono::duration_cast<std::chrono::minutes>(t1 - START_TIME).count();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(t1 - START_TIME).count() % 60;
    std::cout << "\nDurée totale : " << min << "m " << sec << "s\n";

    std::cout.rdbuf(coutbuf);
    log_file.close();

    test_loader.reset();
    model.exportPredictionsToCSV(test_loader, "./logs/predictions_3d_sparse_" + dataset.name + ".csv");

    return 0;
}

// =============================================================================
// Pipeline 3D attention (dense + attention) - Refactorisé
// =============================================================================
static int run3DAttn(bool trainBeforeTesting = true) {
    auto dataset = DatasetManager::getInfo(ACTIVE_DATASET_3D);
    
    section("Pipeline 3D dense + WindowAttention - " + dataset.name);
    requireDir(dataset.path);
    
    std::string filename = "./models/3d_attn_" + dataset.name + ".bin";
    printDatasetInfo(dataset);
    
    MedMNIST3DDataset train_ds(dataset.path, Split::TRAIN, dataset.num_classes, dataset.name, true);
    MedMNIST3DDataset val_ds(dataset.path, Split::VAL, dataset.num_classes, dataset.name, true);
    MedMNIST3DDataset test_ds(dataset.path, Split::TEST, dataset.num_classes, dataset.name, true);
    
    std::cout << "Train : " << train_ds.getNumSamples()
              << "  Val : " << val_ds.getNumSamples() << "\n";
    
    section("Architecture 3D dense + attention");
    CNN model = buildModel3DAttn(dataset.num_classes);
    {
        Tensor probe(1, 1, VOL_SIZE, VOL_SIZE, VOL_SIZE);
        DimensionCalculator::debugArchitecture(model, probe);
    }
    DataLoader3D train_loader(train_ds, BATCH_SIZE_SPARSE, true);
    DataLoader3D val_loader(val_ds, BATCH_SIZE_SPARSE, false);
    DataLoader3D test_loader(test_ds, BATCH_SIZE_SPARSE, false);

    if(AUGMENT_DATA){
        train_loader.setAugmentation();
        std::cout << "\n===> Data Augmented" << std::endl;
    }

    if(MAX_SAMPLE){
        train_loader.setMaxSamples(MAX_SAMPLE_TRAIN); 
        val_loader.setMaxSamples(MAX_SAMPLE_VAL);
        std::cout << "\n===> Max samples : (" << MAX_SAMPLE_TRAIN << ") :Train (" << MAX_SAMPLE_VAL << ") :VAL " << std::endl;
    }

    if (trainBeforeTesting) {
        section("Entraînement 3D + attention");
        EarlyStopping es;
        es.patience = PATIENCE;
        es.min_delta = 5e-4f;
        es.checkpoint = filename;
        es.log_file = "./logs/train_3d_attn_" + dataset.name + ".txt";
        model.fitWithValidation(train_loader, val_loader, EPOCHS_3D, BATCH_SIZE_3D, es);
        model.saveParameters(filename);
    } else {
        if(fs::exists(filename)) model.loadParameters(filename);
    }
    
    section("Évaluation finale sur le Test Set");
    std::string log_filename = "./logs/eval_3d_attn_" + dataset.name + ".txt";
    std::cout << "Saving evaluation results to " << log_filename << "\n";
    requireDir("./logs");
    std::ofstream log_file(log_filename);
    TeeBuffer tee(std::cout.rdbuf(), log_file.rdbuf(), true);
    std::ostream tee_stream(&tee);
    std::streambuf* coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(tee_stream.rdbuf());

    model.evaluate(test_loader);

    test_loader.reset();
    auto cm = model.confusionMatrix(test_loader, dataset.num_classes);
    model.printConfusionMatrix(cm, dataset.class_names);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto min = std::chrono::duration_cast<std::chrono::minutes>(t1 - START_TIME).count();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(t1 - START_TIME).count() % 60;
    std::cout << "\nDurée totale : " << min << "m " << sec << "s\n";

    std::cout.rdbuf(coutbuf);
    log_file.close();

    test_loader.reset();
    model.exportPredictionsToCSV(test_loader, "./logs/predictions_3d_attn_" + dataset.name + ".csv");

    return 0;
}

// =============================================================================
// Pipeline 3D sparse + attention - Refactorisé
// =============================================================================
static int run3DSparseAttn(bool trainBeforeTesting = true) {
    auto dataset = DatasetManager::getInfo(ACTIVE_DATASET_3D);
    
    section("Pipeline 3D sparse + WindowAttention (Flash ST-Attention) - " + dataset.name);
    requireDir(dataset.path);
    
    std::string filename = "./models/3d_sparse_attn_" + dataset.name + ".bin";
    printDatasetInfo(dataset);

    MedMNIST3DDataset train_ds(dataset.path, Split::TRAIN, dataset.num_classes, dataset.name, true);
    MedMNIST3DDataset val_ds(dataset.path, Split::VAL, dataset.num_classes, dataset.name, true);
    MedMNIST3DDataset test_ds(dataset.path, Split::TEST, dataset.num_classes, dataset.name, true);

    std::cout << "Train     : " << train_ds.getNumSamples() << "\n"
              << "Val       : " << val_ds.getNumSamples() << "\n"
              << "Test      : " << test_ds.getNumSamples() << "\n"
              << "Threshold : " << SPARSE_THRESHOLD << "\n";

    section("Architecture 3D sparse + attention");
    CNN model = buildModel3DSparseAttn(dataset.num_classes);
    {
        Tensor probe(1, 1, VOL_SIZE, VOL_SIZE, VOL_SIZE);
        DimensionCalculator::debugArchitecture(model, probe);
    }
    DataLoader3D train_loader(train_ds, BATCH_SIZE_SPARSE, true);
    DataLoader3D val_loader(val_ds, BATCH_SIZE_SPARSE, false);
    DataLoader3D test_loader(test_ds, BATCH_SIZE_SPARSE, false);
   
    if(AUGMENT_DATA){
        train_loader.setAugmentation();
        std::cout << "\n===> Data Augmented" << std::endl;
    }

    if(MAX_SAMPLE){
        train_loader.setMaxSamples(MAX_SAMPLE_TRAIN); 
        val_loader.setMaxSamples(MAX_SAMPLE_VAL);
        std::cout << "\n===> Max samples : (" << MAX_SAMPLE_TRAIN << ") :Train (" << MAX_SAMPLE_VAL << ") :VAL " << std::endl;
    }

    if (trainBeforeTesting) {
        section("Entraînement 3D sparse + attention");
        EarlyStopping es;
        es.patience = PATIENCE;
        es.min_delta = 5e-4f;
        es.restore_best = true;
        es.checkpoint = filename;
        es.log_file = "./logs/train_3d_sparse_attn_" + dataset.name + ".txt";
        model.fitWithValidation(train_loader, val_loader, EPOCHS_SPARSE, BATCH_SIZE_SPARSE, es);
        model.saveParameters(filename);
    } else {
        if(fs::exists(filename)) model.loadParameters(filename);
    }
    
    section("Évaluation finale sur le Test Set");
    std::string log_filename = "./logs/eval_3d_sparse_attn_" + dataset.name + ".txt";
    std::cout << "Saving evaluation results to " << log_filename << "\n";
    requireDir("./logs");
    std::ofstream log_file(log_filename);
    TeeBuffer tee(std::cout.rdbuf(), log_file.rdbuf(), true);
    std::ostream tee_stream(&tee);
    std::streambuf* coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(tee_stream.rdbuf());

    model.evaluate(test_loader);

    test_loader.reset();
    auto cm = model.confusionMatrix(test_loader, dataset.num_classes);
    model.printConfusionMatrix(cm, dataset.class_names);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto min = std::chrono::duration_cast<std::chrono::minutes>(t1 - START_TIME).count();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(t1 - START_TIME).count() % 60;
    std::cout << "\nDurée totale : " << min << "m " << sec << "s\n";

    std::cout.rdbuf(coutbuf);
    log_file.close();

    test_loader.reset();
    model.exportPredictionsToCSV(test_loader, "./logs/predictions_3d_sparse_attn_" + dataset.name + ".csv");

    return 0;
}

// =============================================================================
// Pipeline 3D ConvNeXt Dense - Refactorisé
// =============================================================================
static int run3DConvNeXtDense(bool trainBeforeTesting = true) {
    auto dataset = DatasetManager::getInfo(ACTIVE_DATASET_3D);
    
    section("Pipeline 3D ConvNeXt (Dense) - " + dataset.name);
    requireDir(dataset.path);
    
    std::string filename = "./models/3d_convnext_dense_" + dataset.name + ".bin";
    printDatasetInfo(dataset);

    MedMNIST3DDataset train_ds(dataset.path, Split::TRAIN, dataset.num_classes, dataset.name, true);
    MedMNIST3DDataset val_ds(dataset.path, Split::VAL, dataset.num_classes, dataset.name, true);
    MedMNIST3DDataset test_ds(dataset.path, Split::TEST, dataset.num_classes, dataset.name, true);
    
    CNN model = buildConvNeXt3DDense(dataset.num_classes);
    {
        Tensor probe(1, 1, VOL_SIZE, VOL_SIZE, VOL_SIZE);
        DimensionCalculator::debugArchitecture(model, probe);
    }
    DataLoader3D train_loader(train_ds, BATCH_SIZE_3D, true);
    DataLoader3D val_loader(val_ds, BATCH_SIZE_3D, false);
    DataLoader3D test_loader(test_ds, BATCH_SIZE_3D, false);
    
    if(AUGMENT_DATA){
        train_loader.setAugmentation();
        std::cout << "\n===> Data Augmented" << std::endl;
    }

    if(MAX_SAMPLE){
        train_loader.setMaxSamples(MAX_SAMPLE_TRAIN); 
        val_loader.setMaxSamples(MAX_SAMPLE_VAL);
        std::cout << "\n===> Max samples : (" << MAX_SAMPLE_TRAIN << ") :Train (" << MAX_SAMPLE_VAL << ") :VAL " << std::endl;
    }

    if (trainBeforeTesting) {
        EarlyStopping es;
        es.patience = PATIENCE;
        es.min_delta = 5e-4f;
        es.checkpoint = filename;
        es.log_file = "./logs/train_3d_convnext_dense_" + dataset.name + ".txt";
        model.fitWithValidation(train_loader, val_loader, EPOCHS_3D, BATCH_SIZE_3D, es);
        model.saveParameters(filename);
    } else {
        if(fs::exists(filename)) model.loadParameters(filename);
        else std::cout << "No saved model parameters " << filename << std::endl;
    }
    
    section("Evaluation finale sur le Test Set");
    std::string log_filename = "./logs/eval_3d_convnext_dense_" + dataset.name + ".txt";
    std::cout << "Saving evaluation results to " << log_filename << "\n";
    requireDir("./logs");
    std::ofstream log_file(log_filename);
    TeeBuffer tee(std::cout.rdbuf(), log_file.rdbuf(), true);
    std::ostream tee_stream(&tee);
    std::streambuf* coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(tee_stream.rdbuf());

    model.evaluate(test_loader);

    test_loader.reset();
    auto cm = model.confusionMatrix(test_loader, dataset.num_classes);
    model.printConfusionMatrix(cm, dataset.class_names);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto min = std::chrono::duration_cast<std::chrono::minutes>(t1 - START_TIME).count();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(t1 - START_TIME).count() % 60;
    std::cout << "\nDuree totale : " << min << "m " << sec << "s\n";

    std::cout.rdbuf(coutbuf);
    log_file.close();

    test_loader.reset();
    model.exportPredictionsToCSV(test_loader, "./logs/predictions_3d_convnext_dense_" + dataset.name + ".csv");

    return 0;
}



// =============================================================================
// Pipeline 3D ConvNeXt Sparse - Refactorisé
// =============================================================================
static int run3DConvNeXtSparse(bool trainBeforeTesting = true) {
    auto dataset = DatasetManager::getInfo(ACTIVE_DATASET_3D);
    
    section("Pipeline 3D ConvNeXt (Sparse) - " + dataset.name);
    requireDir(dataset.path);
    
    std::string filename = "./models/3d_convnext_sparse_" + dataset.name + ".bin";
    printDatasetInfo(dataset);

    MedMNIST3DDataset train_ds(dataset.path, Split::TRAIN, dataset.num_classes, dataset.name, true);
    MedMNIST3DDataset val_ds(dataset.path, Split::VAL, dataset.num_classes, dataset.name, true);
    MedMNIST3DDataset test_ds(dataset.path, Split::TEST, dataset.num_classes, dataset.name, true);
    
    CNN model = buildConvNeXt3DSparse(dataset.num_classes);
    {
        Tensor probe(1, 1, VOL_SIZE, VOL_SIZE, VOL_SIZE);
        DimensionCalculator::debugArchitecture(model, probe);
    }
    DataLoader3D train_loader(train_ds, BATCH_SIZE_SPARSE, true);
    DataLoader3D val_loader(val_ds, BATCH_SIZE_SPARSE, false);
    DataLoader3D test_loader(test_ds, BATCH_SIZE_SPARSE, false);

    if(AUGMENT_DATA){
        train_loader.setAugmentation();
        std::cout << "\n===> Data Augmented" << std::endl;
    }

    if(MAX_SAMPLE){
        train_loader.setMaxSamples(MAX_SAMPLE_TRAIN); 
        val_loader.setMaxSamples(MAX_SAMPLE_VAL);
        std::cout << "\n===> Max samples : (" << MAX_SAMPLE_TRAIN << ") :Train (" << MAX_SAMPLE_VAL << ") :VAL " << std::endl;
    }

    if (trainBeforeTesting) {
        EarlyStopping es;
        es.patience = PATIENCE;
        es.min_delta = 5e-4f;
        es.checkpoint = filename;
        es.log_file = "./logs/train_3d_convnext_sparse_" + dataset.name + ".txt";
        model.fitWithValidation(train_loader, val_loader, EPOCHS_SPARSE, BATCH_SIZE_SPARSE, es);
        model.saveParameters(filename);
    } else {
        if(fs::exists(filename)) model.loadParameters(filename);
        else std::cout << "No saved model parameters " << filename << std::endl;
    }
    
    section("Evaluation finale sur le Test Set");
    std::string log_filename = "./logs/eval_3d_convnext_sparse_" + dataset.name + ".txt";
    std::cout << "Saving evaluation results to " << log_filename << "\n";
    requireDir("./logs");
    std::ofstream log_file(log_filename);
    TeeBuffer tee(std::cout.rdbuf(), log_file.rdbuf(), true);
    std::ostream tee_stream(&tee);
    std::streambuf* coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(tee_stream.rdbuf());

    model.evaluate(test_loader);

    test_loader.reset();
    auto cm = model.confusionMatrix(test_loader, dataset.num_classes);
    model.printConfusionMatrix(cm, dataset.class_names);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto min = std::chrono::duration_cast<std::chrono::minutes>(t1 - START_TIME).count();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(t1 - START_TIME).count() % 60;
    std::cout << "\nDuree totale : " << min << "m " << sec << "s\n";

    std::cout.rdbuf(coutbuf);
    log_file.close();

    test_loader.reset();
    model.exportPredictionsToCSV(test_loader, "./logs/predictions_3d_convnext_sparse_" + dataset.name + ".csv");

    return 0;
}

// =============================================================================
// Point d'entrée
// =============================================================================
int main(int argc, char* argv[]) {
    try {
        std::string arg;
        bool trainBeforeTesting = true;
        
        if (argc > 1) {
            arg = argv[1];
            if (arg == "2d") { ACTIVE_PIPELINE = Pipeline::CNN2D; }
            else if (arg == "3d") { ACTIVE_PIPELINE = Pipeline::CNN3D; }
            else if (arg == "sparse") { ACTIVE_PIPELINE = Pipeline::CNN3D_SPARSE; }
            else if (arg == "attn") { ACTIVE_PIPELINE = Pipeline::CNN3D_ATTN; }
            else if (arg == "sparse_attn") { ACTIVE_PIPELINE = Pipeline::CNN3D_SPARSE_ATTN; }
            else if (arg == "convnext_dense") { ACTIVE_PIPELINE = Pipeline::CNN3D_CONVNEXT_DENSE; }
            else if (arg == "convnext_sparse") { ACTIVE_PIPELINE = Pipeline::CNN3D_CONVNEXT_SPARSE; }
            else {
                std::cerr << "Usage: " << argv[0] << " [2d|3d|sparse|attn|sparse_attn|convnext_dense|convnext_sparse] [--skip-train]\n";
                return 1;
            }
        }
        
        for (int i = 1; i < argc; ++i) {
            std::string flag = argv[i];
            if (flag == "--skip-train") {
                trainBeforeTesting = false;
                break;
            }
        }

        Eigen::initParallel();
        Eigen::setNbThreads(4);

        START_TIME = std::chrono::high_resolution_clock::now();

        int ret = 0;
        switch (ACTIVE_PIPELINE) {
        case Pipeline::CNN2D: ret = run2D(trainBeforeTesting); break;
        case Pipeline::CNN3D: ret = run3D(trainBeforeTesting); break;
        case Pipeline::CNN3D_SPARSE: ret = run3DSparse(trainBeforeTesting); break;
        case Pipeline::CNN3D_ATTN: ret = run3DAttn(trainBeforeTesting); break;
        case Pipeline::CNN3D_SPARSE_ATTN: ret = run3DSparseAttn(trainBeforeTesting); break;
        case Pipeline::CNN3D_CONVNEXT_DENSE: ret = run3DConvNeXtDense(trainBeforeTesting); break;
        case Pipeline::CNN3D_CONVNEXT_SPARSE: ret = run3DConvNeXtSparse(trainBeforeTesting); break;
        }

        return ret;
    }
    catch (const std::exception& e) {
        std::cerr << "\n[ERREUR] " << e.what() << "\n";
        return 1;
    }
}