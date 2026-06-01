// main.cpp
#include "shared.hpp"
#include "CNNLIB.hpp"
#include "ModelBuilders.hpp"

enum class Pipeline { CNN2D, CNN3D, CNN3D_SPARSE, CNN3D_ATTN, CNN3D_SPARSE_ATTN, CNN3D_CONVNEXT_DENSE, CNN3D_CONVNEXT_SPARSE };
static Pipeline ACTIVE_PIPELINE = Pipeline::CNN2D;
static std::chrono::high_resolution_clock::time_point START_TIME;


#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <fstream>

// =============================================================================
// Pipeline 2D — inchangé
// =============================================================================



static int run2D(bool trainBeforeTesting = true) {
    section("Pipeline 2D");
    const std::string train_dir = USE_BLOODCELLS ? BLOODCELLS_TRAIN_PATH : MNIST_TRAIN_PATH;
    const std::string test_dir = USE_BLOODCELLS ? BLOODCELLS_TEST_PATH : MNIST_TEST_PATH;
    requireDir(train_dir); requireDir(test_dir);
    ImageFolderDataset train_ds(train_dir, IMAGE_SIZE_2D, IMAGE_SIZE_2D, true, true);
    ImageFolderDataset val_ds(test_dir, IMAGE_SIZE_2D, IMAGE_SIZE_2D, false, true);
    CNN model = buildModel2D(train_ds.getNumClasses());
    DataLoader train_loader(train_ds, BATCH_SIZE_2D, true);
    DataLoader val_loader(val_ds, BATCH_SIZE_2D, false);
    // train_loader.setMaxSamples(200); 
    // val_loader.setMaxSamples(100);
    
    std::string filename = "./models/2d_best.bin";
    if (trainBeforeTesting) {
        EarlyStopping es;
        es.patience = 20;
        es.min_delta = 5e-4f;
        es.checkpoint = filename;
        es.log_file = "./logs/train_2d.txt";
        model.fitWithValidation(train_loader, val_loader, EPOCHS_2D, BATCH_SIZE_2D, es);
        model.saveParameters(filename);
    } else {
        if(fs::exists(filename))model.loadParameters(filename);
    }

    section("Évaluation finale sur le Test Set");
    std::cout << "Saving evaluation results to ./logs/eval_2d.txt\n";
    requireDir("./logs");
    std::ofstream log_file("./logs/eval_2d.txt");
    TeeBuffer tee(std::cout.rdbuf(), log_file.rdbuf(), true);
    std::ostream tee_stream(&tee);
    std::streambuf* coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(tee_stream.rdbuf());
    
    model.evaluate(val_loader);

    val_loader.reset();
    auto cm = model.confusionMatrix(val_loader, train_ds.getNumClasses());
    model.printConfusionMatrix(cm);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto min = std::chrono::duration_cast<std::chrono::minutes>(t1 - START_TIME).count();
    auto sec = std::chrono::duration_cast<std::chrono::seconds> (t1 - START_TIME).count() % 60;
    std::cout << "\nDurée totale : " << min << "m " << sec << "s\n";

    std::cout.rdbuf(coutbuf);
    log_file.close();

    val_loader.reset();
    model.exportPredictionsToCSV(val_loader, "./logs/predictions_2d.csv");

    return 0;
}

// =============================================================================
// Pipeline 3D dense — inchangé
// =============================================================================



static int run3D(bool trainBeforeTesting = true) {
    section("Pipeline 3D dense");
    requireDir(FRACTURE_PATH);
    std::string filename = "./models/3d_best.bin";

    MedMNIST3DDataset train_ds(FRACTURE_PATH, Split::TRAIN, 3, "FractureMNIST3D", true);
    MedMNIST3DDataset val_ds(FRACTURE_PATH, Split::VAL, 3, "FractureMNIST3D", true);
    MedMNIST3DDataset test_ds(FRACTURE_PATH, Split::TEST, 3, "FractureMNIST3D", true);
    CNN model = buildModel3D(3);
    
    DataLoader3D train_loader(train_ds, BATCH_SIZE_3D, true);
    DataLoader3D val_loader(val_ds, BATCH_SIZE_3D, false);
    DataLoader3D test_loader(test_ds, BATCH_SIZE_3D, false);
    // train_loader.setAugmentation();  // augmentation entraînement uniquement
    // train_loader.setMaxSamples(32); val_loader.setMaxSamples(16);
    
    if (trainBeforeTesting) {
        EarlyStopping es;
        es.patience = 20;
        es.min_delta = 5e-4f;
        es.checkpoint = filename;
        es.log_file = "./logs/train_3d.txt";
        model.fitWithValidation(train_loader, val_loader, EPOCHS_3D, BATCH_SIZE_3D, es);
        model.saveParameters(filename);
    } else {
        if(fs::exists(filename))model.loadParameters(filename);
    }
    
    section("Évaluation finale sur le Test Set");
    std::cout << "Saving evaluation results to ./logs/eval_3d.txt\n";
    requireDir("./logs");
    std::ofstream log_file("./logs/eval_3d.txt");
    TeeBuffer tee(std::cout.rdbuf(), log_file.rdbuf(), true);
    std::ostream tee_stream(&tee);
    std::streambuf* coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(tee_stream.rdbuf());

    model.evaluate(test_loader);

    // Matrice de confusion
    test_loader.reset();
    auto cm = model.confusionMatrix(test_loader, 3);
    model.printConfusionMatrix(cm, {"No fracture", "Fracture T1", "Fracture T2"});

    auto t1 = std::chrono::high_resolution_clock::now();
    auto min = std::chrono::duration_cast<std::chrono::minutes>(t1 - START_TIME).count();
    auto sec = std::chrono::duration_cast<std::chrono::seconds> (t1 - START_TIME).count() % 60;
    std::cout << "\nDurée totale : " << min << "m " << sec << "s\n";

    std::cout.rdbuf(coutbuf);
    log_file.close();

    test_loader.reset();
    model.exportPredictionsToCSV(test_loader, "./logs/predictions_3d.csv");

    return 0;
}

// =============================================================================
// Pipeline 3D sparse — sans attention (référence)
// =============================================================================



static int run3DSparse(bool trainBeforeTesting = true) {
    section("Pipeline 3D sparse (sans attention)");
    std::string filename = "./models/sparse_best.bin";
    requireDir(FRACTURE_PATH);
    MedMNIST3DDataset train_ds(FRACTURE_PATH, Split::TRAIN, 3, "AdrenalMNIST3D", true);
    MedMNIST3DDataset val_ds(FRACTURE_PATH, Split::VAL, 3, "AdrenalMNIST3D", true);
    MedMNIST3DDataset test_ds(FRACTURE_PATH, Split::TEST, 3, "AdrenalMNIST3D", true);
    CNN model = buildModel3DSparse(3);

    DataLoader3D train_loader(train_ds, BATCH_SIZE_SPARSE, true);
    DataLoader3D val_loader(val_ds, BATCH_SIZE_SPARSE, false);
    DataLoader3D test_loader(test_ds, BATCH_SIZE_SPARSE, false);

    train_loader.setAugmentation();  // augmentation entraînement uniquement
    // train_loader.setMaxSamples(714);
    // val_loader.setMaxSamples(102);
    // test_loader.setMaxSamples(204);

    if (trainBeforeTesting) {
        EarlyStopping es;
        es.patience = 20;
        es.min_delta = 5e-4f;
        es.checkpoint = filename;
        es.log_file = "./logs/train_sparse.txt";
        model.fitWithValidation(train_loader, val_loader, EPOCHS_SPARSE, BATCH_SIZE_SPARSE, es);
        model.saveParameters(filename);
    } else {
        if(fs::exists(filename))model.loadParameters(filename);
    }
    
    section("Évaluation finale sur le Test Set");
    std::cout << "Saving evaluation results to ./logs/eval_sparse.txt\n";
    requireDir("./logs");
    std::ofstream log_file("./logs/eval_sparse.txt");
    TeeBuffer tee(std::cout.rdbuf(), log_file.rdbuf(), true);
    std::ostream tee_stream(&tee);
    std::streambuf* coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(tee_stream.rdbuf());

    model.evaluate(test_loader);

    // Matrice de confusion
    test_loader.reset();  // important — evaluate() a consommé le loader
    auto cm = model.confusionMatrix(test_loader, 3);
    model.printConfusionMatrix(cm, {"No fracture", "Fracture T1", "Fracture T2"});

    auto t1 = std::chrono::high_resolution_clock::now();
    auto min = std::chrono::duration_cast<std::chrono::minutes>(t1 - START_TIME).count();
    auto sec = std::chrono::duration_cast<std::chrono::seconds> (t1 - START_TIME).count() % 60;
    std::cout << "\nDurée totale : " << min << "m " << sec << "s\n";

    std::cout.rdbuf(coutbuf);
    log_file.close();

    test_loader.reset();
    model.exportPredictionsToCSV(test_loader, "./logs/predictions_sparse.csv");

    return 0;
}

// =============================================================================
// Pipeline 3D dense + WindowAttention
// =============================================================================
//
// Architecture :
//
//   ConvLayer3D 1→16  s=1   (B, 16, 28, 28, 28)
//   ReLU
//   ConvLayer3D 16→32 s=2   (B, 32, 14, 14, 14)
//   ReLU
//   WindowAttention3D C=32, win=4×4×4, 4 têtes   ← après bloc 2
//     Découpe 14³ en fenêtres 4³ → 4×4×4 = 8 fenêtres
//     Matrice attention : 64×64 par fenêtre      (légère)
//     Captule les relations spatiales à l'échelle 14³
//   ConvLayer3D 32→64 s=2   (B, 64, 7, 7, 7)
//   ReLU
//   WindowAttention3D C=64, win=7×7×7, 4 têtes   ← après bloc 3
//     Une seule fenêtre couvre tout le volume 7³
//     Matrice attention : 343×343                (raisonnable)
//     Attention globale sur la représentation finale
//   GlobalAvgPool3D             (B, 64, 1, 1, 1)
//   Dense 64→256 + ReLU + Dropout + Dense 256→3
//   SoftmaxCE
//
// =============================================================================



static int run3DAttn(bool trainBeforeTesting = true) {
    section("Pipeline 3D dense + WindowAttention");
    requireDir(FRACTURE_PATH);
    std::string filename = "./models/3d_attn_best.bin";
    MedMNIST3DDataset train_ds(FRACTURE_PATH, Split::TRAIN, 3, "FractureMNIST3D", true);
    MedMNIST3DDataset val_ds(FRACTURE_PATH, Split::VAL, 3, "FractureMNIST3D", true);
    MedMNIST3DDataset test_ds(FRACTURE_PATH, Split::TEST, 3, "FractureMNIST3D", true);
    
    std::cout << "Train : " << train_ds.getNumSamples()
        << "  Val : " << val_ds.getNumSamples() << "\n";
    section("Architecture 3D dense + attention");
    CNN model = buildModel3DAttn(3);
    {
        Tensor probe(1, 1, VOL_SIZE, VOL_SIZE, VOL_SIZE);
        DimensionCalculator::debugArchitecture(model, probe);
    }
    DataLoader3D train_loader(train_ds, BATCH_SIZE_SPARSE, true);
    DataLoader3D val_loader(val_ds, BATCH_SIZE_SPARSE, false);
    DataLoader3D test_loader(test_ds, BATCH_SIZE_SPARSE, false);
    // train_loader.setAugmentation();  // augmentation entraînement uniquement
    // train_loader.setMaxSamples(32); val_loader.setMaxSamples(16);
    train_loader.setMaxSamples(714); //714 échantillons d'entraînement → 45 batches par époque
    val_loader.setMaxSamples(102); //102 échantillons de validation → 6-7 batches par époque
    test_loader.setMaxSamples(204);//204 échantillons de test → 12-13 batches pour l'évaluation finale

    if (trainBeforeTesting) {
        section("Entraînement 3D + attention");
        EarlyStopping es;
        es.patience = 20;
        es.min_delta = 5e-4f;
        es.checkpoint = filename;
        es.log_file = "./logs/train_3d_attn.txt";
        model.fitWithValidation(train_loader, val_loader, EPOCHS_3D, BATCH_SIZE_3D, es);
        model.saveParameters(filename);
    } else {
        if(fs::exists(filename))model.loadParameters(filename);
    }
    
    section("Évaluation finale sur le Test Set");
    std::cout << "Saving evaluation results to ./logs/eval_3d_attn.txt\n";
    requireDir("./logs");
    std::ofstream log_file("./logs/eval_3d_attn.txt");
    TeeBuffer tee(std::cout.rdbuf(), log_file.rdbuf(), true);
    std::ostream tee_stream(&tee);
    std::streambuf* coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(tee_stream.rdbuf());

    model.evaluate(test_loader);

    // Matrice de confusion
    test_loader.reset();
    auto cm = model.confusionMatrix(test_loader, 3);
    model.printConfusionMatrix(cm, {"No fracture", "Fracture T1", "Fracture T2"});

    auto t1 = std::chrono::high_resolution_clock::now();
    auto min = std::chrono::duration_cast<std::chrono::minutes>(t1 - START_TIME).count();
    auto sec = std::chrono::duration_cast<std::chrono::seconds> (t1 - START_TIME).count() % 60;
    std::cout << "\nDurée totale : " << min << "m " << sec << "s\n";

    std::cout.rdbuf(coutbuf);
    log_file.close();

    test_loader.reset();
    model.exportPredictionsToCSV(test_loader, "./logs/predictions_3d_attn.csv");

    return 0;
}

// =============================================================================
// Pipeline 3D sparse + WindowAttention  ← pipeline principal
// =============================================================================
//
// Architecture :
//
//   SparseConvAdapter 1→16  SubManifold s=1    (B, 16, 28, 28, 28)
//   ReLU
//   SparseConvAdapter 16→32 Standard   s=2    (B, 32, 14, 14, 14)
//   ReLU
//   WindowAttention3D C=32, win=4, heads=4    ← attention sur 14³
//     Opère sur Tensor dense produit par to_dense()
//     Captule les dépendances spatiales entre voxels osseux à 14³
//   SparseConvAdapter 32→64 Standard   s=2    (B, 64, 7, 7, 7)
//   ReLU
//   WindowAttention3D C=64, win=7, heads=4    ← attention globale sur 7³
//     Fenêtre unique = attention sur toute la représentation finale
//     343 tokens × 64 canaux : représentation très compacte
//   SparseGlobalAvgPool                       (B, 64, 1, 1, 1)
//   Dense 64→256 + ReLU + Dropout + Dense 256→3
//   SoftmaxCE
//
// Paramètres totaux :
//   SparseConv : 448 + 13 856 + 55 360          =  69 664
//   Attention  : 2×(4×32²) + 2×(4×64²) + norms =  50 432
//   Classif.   : 16 640 + 771                   =  17 411
//   TOTAL                                       = 137 507
//
// =============================================================================


static int run3DSparseAttn(bool trainBeforeTesting = true) {
    section("Pipeline 3D sparse + WindowAttention (Flash ST-Attention)");
    std::string filename = "./models/sparse_attn_best.bin";
    requireDir(FRACTURE_PATH);

    MedMNIST3DDataset train_ds(FRACTURE_PATH, Split::TRAIN, 3, "FractureMNIST3D", true);
    MedMNIST3DDataset val_ds(FRACTURE_PATH, Split::VAL, 3, "FractureMNIST3D", true);
    MedMNIST3DDataset test_ds(FRACTURE_PATH, Split::TEST, 3, "FractureMNIST3D", true);

    std::cout << "Classes   : 3\n"
        << "Train     : " << train_ds.getNumSamples() << "\n"
        << "Val       : " << val_ds.getNumSamples() << "\n"
        << "Test      : " << test_ds.getNumSamples() << "\n"
        << "Threshold : " << SPARSE_THRESHOLD << "\n";

    section("Architecture 3D sparse + attention");
    CNN model = buildModel3DSparseAttn(3);
    // debugArchitecture fonctionne car WindowAttention3DLayer hérite de Layer
    {
        Tensor probe(1, 1, VOL_SIZE, VOL_SIZE, VOL_SIZE);
        DimensionCalculator::debugArchitecture(model, probe);
    }

    // Affichage du nombre de paramètres des couches d'attention
    std::cout << "\nParamètres attention :\n";
    for (int i = 0; i < static_cast<int>(model.getLayers().size()); ++i) {
        auto* attn = dynamic_cast<WindowAttention3DLayer*>(model.getLayer(i));
        if (attn) {
            std::cout << "  Layer " << std::setw(2) << i
                << " [" << attn->getName() << "]"
                << "  params=" << attn->numParams() << "\n";
        }
    }

    DataLoader3D train_loader(train_ds, BATCH_SIZE_SPARSE, true);
    DataLoader3D val_loader(val_ds, BATCH_SIZE_SPARSE, false);
    DataLoader3D test_loader(test_ds, BATCH_SIZE_SPARSE, false);
   
    train_loader.setAugmentation();  // augmentation entraînement uniquement
    // train_loader.setMaxSamples(14); // dataset complet
    // val_loader.setMaxSamples(2);   // dataset complet

    if (trainBeforeTesting) {
        section("Entraînement 3D sparse + attention");
        EarlyStopping es;
        es.patience    = 20;
        es.min_delta   = 5e-4f;
        es.restore_best = true;
        es.checkpoint  = filename;
        es.log_file    = "./logs/train_sparse_attn.txt";
        model.fitWithValidation(train_loader, val_loader, EPOCHS_SPARSE, BATCH_SIZE_SPARSE, es);
        model.saveParameters(filename);
    } else {
        if(fs::exists(filename))model.loadParameters(filename);
    }
    
    section("Évaluation finale sur le Test Set");
    std::cout << "Saving evaluation results to ./logs/eval_sparse_attn.txt\n";
    requireDir("./logs");
    std::ofstream log_file("./logs/eval_sparse_attn.txt");
    TeeBuffer tee(std::cout.rdbuf(), log_file.rdbuf(), true);
    std::ostream tee_stream(&tee);
    std::streambuf* coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(tee_stream.rdbuf());

    model.evaluate(test_loader);

    // Matrice de confusion
    test_loader.reset();
    auto cm = model.confusionMatrix(test_loader, 3);
    model.printConfusionMatrix(cm, {"No fracture", "Fracture T1", "Fracture T2"});

    auto t1 = std::chrono::high_resolution_clock::now();
    auto min = std::chrono::duration_cast<std::chrono::minutes>(t1 - START_TIME).count();
    auto sec = std::chrono::duration_cast<std::chrono::seconds> (t1 - START_TIME).count() % 60;
    std::cout << "\nDurée totale : " << min << "m " << sec << "s\n";

    std::cout.rdbuf(coutbuf);
    log_file.close();

    // Export pour courbes ROC/AUC
    test_loader.reset();
    model.exportPredictionsToCSV(test_loader, "./logs/predictions_sparse_attn.csv");

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
            else {
                std::cerr << "Usage: " << argv[0] << " [2d|3d|sparse|attn|sparse_attn] [--skip-train]\n";
                return 1;
            }
        }
        
        // Parse --skip-train flag
        for (int i = 1; i < argc; ++i) {
            std::string flag = argv[i];
            if (flag == "--skip-train") {
                trainBeforeTesting = false;
                break;
            }
        }

        // We don't want to parallelize with eigen
        Eigen::initParallel();
        Eigen::setNbThreads(4);

        START_TIME = std::chrono::high_resolution_clock::now();

        int ret = 0;
        switch (ACTIVE_PIPELINE) {
        case Pipeline::CNN2D:
            ret = run2D(trainBeforeTesting);           break;
        case Pipeline::CNN3D:
            ret = run3D(trainBeforeTesting);           break;
        case Pipeline::CNN3D_SPARSE:
            ret = run3DSparse(trainBeforeTesting);     break;
        case Pipeline::CNN3D_ATTN:
            ret = run3DAttn(trainBeforeTesting);       break;
        case Pipeline::CNN3D_SPARSE_ATTN:
            ret = run3DSparseAttn(trainBeforeTesting); break;
        }

        return ret;

    }
    catch (const std::exception& e) {
        std::cerr << "\n[ERREUR] " << e.what() << "\n";
        return 1;
    }
}

// =============================================================================
// Pipeline 3D ConvNeXt (Dense)
// =============================================================================
static int run3DConvNeXtDense(bool trainBeforeTesting = true) {
    section("Pipeline 3D ConvNeXt (Dense)");
    requireDir(FRACTURE_PATH);
    std::string filename = "./models/3d_convnext_dense_best.bin";

    MedMNIST3DDataset train_ds(FRACTURE_PATH, Split::TRAIN, 3, "FractureMNIST3D", true);
    MedMNIST3DDataset val_ds(FRACTURE_PATH, Split::VAL, 3, "FractureMNIST3D", true);
    MedMNIST3DDataset test_ds(FRACTURE_PATH, Split::TEST, 3, "FractureMNIST3D", true);
    
    CNN model = buildConvNeXt3DDense(3);
    
    {
        Tensor probe(1, 1, VOL_SIZE, VOL_SIZE, VOL_SIZE);
        DimensionCalculator::debugArchitecture(model, probe);
    }
    
    DataLoader3D train_loader(train_ds, BATCH_SIZE_3D, true);
    DataLoader3D val_loader(val_ds, BATCH_SIZE_3D, false);
    DataLoader3D test_loader(test_ds, BATCH_SIZE_3D, false);
    
    if (trainBeforeTesting) {
        EarlyStopping es;
        es.patience = 20;
        es.min_delta = 5e-4f;
        es.checkpoint = filename;
        es.log_file = "./logs/train_3d_convnext_dense.txt";
        model.fitWithValidation(train_loader, val_loader, EPOCHS_3D, BATCH_SIZE_3D, es);
        model.saveParameters(filename);
    } else {
        if(fs::exists(filename))model.loadParameters(filename);
        else std::cout << "No saved model parameters " << filename << std::endl;
    }
    
    section("Evaluation finale sur le Test Set");
    std::cout << "Saving evaluation results to ./logs/eval_3d_convnext_dense.txt\n";
    requireDir("./logs");
    std::ofstream log_file("./logs/eval_3d_convnext_dense.txt");
    TeeBuffer tee(std::cout.rdbuf(), log_file.rdbuf(), true);
    std::ostream tee_stream(&tee);
    std::streambuf* coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(tee_stream.rdbuf());

    model.evaluate(test_loader);

    test_loader.reset();
    auto cm = model.confusionMatrix(test_loader, 3);
    model.printConfusionMatrix(cm, {"No fracture", "Fracture T1", "Fracture T2"});

    auto t1 = std::chrono::high_resolution_clock::now();
    auto min = std::chrono::duration_cast<std::chrono::minutes>(t1 - START_TIME).count();
    auto sec = std::chrono::duration_cast<std::chrono::seconds> (t1 - START_TIME).count() % 60;
    std::cout << "\nDuree totale : " << min << "m " << sec << "s\n";

    std::cout.rdbuf(coutbuf);
    log_file.close();

    test_loader.reset();
    model.exportPredictionsToCSV(test_loader, "./logs/predictions_3d_convnext_dense.csv");

    return 0;
}

// =============================================================================
// Pipeline 3D ConvNeXt (Sparse)
// =============================================================================
static int run3DConvNeXtSparse(bool trainBeforeTesting = true) {
    section("Pipeline 3D ConvNeXt (Sparse)");
    requireDir(FRACTURE_PATH);
    std::string filename = "./models/3d_convnext_sparse_best.bin";

    MedMNIST3DDataset train_ds(FRACTURE_PATH, Split::TRAIN, 3, "FractureMNIST3D", true);
    MedMNIST3DDataset val_ds(FRACTURE_PATH, Split::VAL, 3, "FractureMNIST3D", true);
    MedMNIST3DDataset test_ds(FRACTURE_PATH, Split::TEST, 3, "FractureMNIST3D", true);
    
    CNN model = buildConvNeXt3DSparse(3);
    
    {
        Tensor probe(1, 1, VOL_SIZE, VOL_SIZE, VOL_SIZE);
        DimensionCalculator::debugArchitecture(model, probe);
    }
    
    DataLoader3D train_loader(train_ds, BATCH_SIZE_SPARSE, true);
    DataLoader3D val_loader(val_ds, BATCH_SIZE_SPARSE, false);
    DataLoader3D test_loader(test_ds, BATCH_SIZE_SPARSE, false);
    
    if (trainBeforeTesting) {
        EarlyStopping es;
        es.patience = 20;
        es.min_delta = 5e-4f;
        es.checkpoint = filename;
        es.log_file = "./logs/train_3d_convnext_sparse.txt";
        model.fitWithValidation(train_loader, val_loader, EPOCHS_SPARSE, BATCH_SIZE_SPARSE, es);
        model.saveParameters(filename);
    } else {
        if(fs::exists(filename))model.loadParameters(filename);
        else std::cout << "No saved model parameters " << filename << std::endl;
    }
    
    section("Evaluation finale sur le Test Set");
    std::cout << "Saving evaluation results to ./logs/eval_3d_convnext_sparse.txt\n";
    requireDir("./logs");
    std::ofstream log_file("./logs/eval_3d_convnext_sparse.txt");
    TeeBuffer tee(std::cout.rdbuf(), log_file.rdbuf(), true);
    std::ostream tee_stream(&tee);
    std::streambuf* coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(tee_stream.rdbuf());

    model.evaluate(test_loader);

    test_loader.reset();
    auto cm = model.confusionMatrix(test_loader, 3);
    model.printConfusionMatrix(cm, {"No fracture", "Fracture T1", "Fracture T2"});

    auto t1 = std::chrono::high_resolution_clock::now();
    auto min = std::chrono::duration_cast<std::chrono::minutes>(t1 - START_TIME).count();
    auto sec = std::chrono::duration_cast<std::chrono::seconds> (t1 - START_TIME).count() % 60;
    std::cout << "\nDuree totale : " << min << "m " << sec << "s\n";

    std::cout.rdbuf(coutbuf);
    log_file.close();

    test_loader.reset();
    model.exportPredictionsToCSV(test_loader, "./logs/predictions_3d_convnext_sparse.csv");

    return 0;
}

// =============================================================================
// Point d'entree
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
