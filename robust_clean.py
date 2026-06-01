import re

with open('src/main.cpp', 'r') as f:
    text = f.read()

# 1. Remove the static CNN build* functions. We know they end with 'return model;\n}'
text = re.sub(r'static CNN build[A-Za-z0-9_]*\([^)]*\)\s*\{.*?return model;\s*\}', '', text, flags=re.DOTALL)

# 2. Add #include "ModelBuilders.hpp"
idx = text.find('#include "CNNLIB.hpp"')
if idx != -1:
    idx += len('#include "CNNLIB.hpp"\n')
    text = text[:idx] + '#include "ModelBuilders.hpp"\n' + text[idx:]

# 3. Handle the configuration block. We want to move START_TIME and Pipeline and configurations to the top.
# The configurations are between Configuration globale and Pipeline 2D
cfg_start = text.find('// Configuration globale')
if cfg_start != -1:
    cfg_start = text.rfind('// =', 0, cfg_start)
    cfg_end = text.find('// Pipeline 2D')
    if cfg_end != -1:
        cfg_end = text.rfind('// =', 0, cfg_end)
        
        # the entire config block is here
        config_block = text[cfg_start:cfg_end]
        
        # We delete it from its current location
        text = text[:cfg_start] + text[cfg_end:]
        
        # Now we extract ONLY START_TIME and Pipeline stuff to inject at the top because ModeBuilders.hpp already has the constants
        # Wait, the constants are already in ModelBuilders.hpp, so we just need ACTIVE_PIPELINE and START_TIME declarations near the top.
        decls = "enum class Pipeline { CNN2D, CNN3D, CNN3D_SPARSE, CNN3D_ATTN, CNN3D_SPARSE_ATTN, CNN3D_CONVNEXT_DENSE, CNN3D_CONVNEXT_SPARSE };\nstatic Pipeline ACTIVE_PIPELINE = Pipeline::CNN2D;\nstatic std::chrono::high_resolution_clock::time_point START_TIME;\n"
        
        # Add decls right after includes
        idx = text.find('#include "ModelBuilders.hpp"\n')
        if idx != -1:
            idx += len('#include "ModelBuilders.hpp"\n')
            text = text[:idx] + "\n" + decls + "\n" + text[idx:]

# 4. Now modify main() to handle the new enums and missing run3DConvNeXt loops.
# Let's delete the old main() and the old Pipeline enum that was just before it.
main_start = text.find('enum class Pipeline')
if main_start != -1:
    main_start = text.rfind('// =================================', 0, main_start)
    if main_start != -1:
        text = text[:main_start]

# 5. Append runs and new main
convnext_runs = """
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
    std::cout << "Saving evaluation results to ./logs/eval_3d_convnext_dense.txt\\n";
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
    std::cout << "\\nDuree totale : " << min << "m " << sec << "s\\n";

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
    std::cout << "Saving evaluation results to ./logs/eval_3d_convnext_sparse.txt\\n";
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
    std::cout << "\\nDuree totale : " << min << "m " << sec << "s\\n";

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
                std::cerr << "Usage: " << argv[0] << " [2d|3d|sparse|attn|sparse_attn|convnext_dense|convnext_sparse] [--skip-train]\\n";
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
        std::cerr << "\\n[ERREUR] " << e.what() << "\\n";
        return 1;
    }
}
"""

with open('src/main.cpp', 'w') as f:
    f.write(text + "\n" + convnext_runs)

print("success")
