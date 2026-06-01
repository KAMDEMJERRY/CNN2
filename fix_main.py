import sys

with open('src/main.cpp', 'r') as f:
    lines = f.readlines()

def remove_ranges(lines, ranges):
    keep = []
    for i, line in enumerate(lines):
        line_num = i + 1
        delete = False
        for start, end in ranges:
            if start <= line_num <= end:
                delete = True
                break
        if not delete:
            keep.append(line)
    return keep

# The ranges to remove from the COMMITTED main.cpp
ranges_to_remove = [
    (11, 54),     # Configs
    (59, 80),     # 2D
    (141, 164),   # 3D
    (228, 258),   # 3D Sparse
    (346, 403),   # 3D Attn
    (503, 549),   # 3D Sparse Attn
    (638, 695)    # main() and its comment block
]

new_lines = remove_ranges(lines, ranges_to_remove)

for i, line in enumerate(new_lines):
    if '#include "CNNLIB.hpp"' in line:
        new_lines.insert(i + 1, '#include "ModelBuilders.hpp"\n')
        break

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
    
    section("\u00c9valuation finale sur le Test Set");
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
    std::cout << "\\nDur\u00e9e totale : " << min << "m " << sec << "s\\n";

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
    
    section("\u00c9valuation finale sur le Test Set");
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
    std::cout << "\\nDur\u00e9e totale : " << min << "m " << sec << "s\\n";

    std::cout.rdbuf(coutbuf);
    log_file.close();

    test_loader.reset();
    model.exportPredictionsToCSV(test_loader, "./logs/predictions_3d_convnext_sparse.csv");

    return 0;
}

// =============================================================================
// Point d'entr\u00e9e
// =============================================================================

enum class Pipeline { CNN2D, CNN3D, CNN3D_SPARSE, CNN3D_ATTN, CNN3D_SPARSE_ATTN, CNN3D_CONVNEXT_DENSE, CNN3D_CONVNEXT_SPARSE };
static Pipeline ACTIVE_PIPELINE = Pipeline::CNN2D;
static std::chrono::high_resolution_clock::time_point START_TIME;

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
    f.writelines(new_lines)
    f.write(convnext_runs)

print("success")
