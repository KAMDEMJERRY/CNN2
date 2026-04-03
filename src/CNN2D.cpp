# include "shared.hpp"
# include "CNNLIB.hpp"
# include <iostream>
# include <iomanip>
# include <chrono>
# include <filesystem>
# include <timer.hpp>

// --- 2D ---
# define MNIST_TEST_PATH    "../dataset/mnist_img/trainingSet/trainingSet/" 
# define MNIST_TRAIN_PATH    "../dataset/mnist_img/trainingSample/trainingSample/"
# define BLOODCELLS_TRAIN_PATH "../dataset/bloodcell/images/TRAIN/"
# define BLOODCELLS_TEST_PATH  "../dataset/bloodcell/images/TEST/"
# define RESULT_FILE "../CNN2_experiments/results/results.csv"

static constexpr bool  USE_BLOODCELLS = false;
static constexpr int   IMAGE_SIZE_2D = 28;
static constexpr int   BATCH_SIZE_2D = 100;
static constexpr int   EPOCHS_2D = 50;
static constexpr float LR_2D = 0.0001f;



// =============================================================================
// Pipeline 2D — inchangé
// =============================================================================

static CNN buildModel2D(int num_classes) {
    CNN model;
    model.addLayer(std::make_shared<ConvLayer>(1, 32, 3, 3, 1, 1, 1, 1));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<ConvLayer>(32, 32, 3, 3, 1, 1, 1, 1));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<MaxPoolLayer>(2, 2));
    model.addLayer(std::make_shared<ConvLayer>(32, 64, 3, 3, 1, 1, 1, 1));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<ConvLayer>(64, 64, 3, 3, 1, 1, 1, 1));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<MaxPoolLayer>(2, 2));
    model.addLayer(std::make_shared<DenseLayer>(7 * 7 * 64, 128));
    model.addLayer(std::make_shared<ReLULayer>());
    model.addLayer(std::make_shared<DropoutLayer>(0.5f));
    model.addLayer(std::make_shared<DenseLayer>(128, num_classes));
    model.setLossLayer(std::make_shared<SoftmaxCrossEntropyLayer>());
    model.setOptimizer(std::make_shared<Adam>(LR_2D));
    return model;
}

static int run2D() {
    section("Pipeline 2D");
    const std::string train_dir = USE_BLOODCELLS ? BLOODCELLS_TRAIN_PATH : MNIST_TRAIN_PATH;
    const std::string test_dir = USE_BLOODCELLS ? BLOODCELLS_TEST_PATH : MNIST_TEST_PATH;
    requireDir(train_dir); requireDir(test_dir);
    ImageFolderDataset train_ds(train_dir, IMAGE_SIZE_2D, IMAGE_SIZE_2D, true, true);
    ImageFolderDataset val_ds(test_dir, IMAGE_SIZE_2D, IMAGE_SIZE_2D, false, true);
    CNN model = buildModel2D(train_ds.getNumClasses());

    DataLoader train_loader(train_ds, BATCH_SIZE_2D, true);
    DataLoader val_loader(val_ds, BATCH_SIZE_2D, false);
    train_loader.setMaxSamples(200); val_loader.setMaxSamples(100);
    model.fitWithValidation(train_loader, val_loader, EPOCHS_2D, BATCH_SIZE_2D);
    // model.fit(val_loader, EPOCHS_2D, BATCH_SIZE_2D);
    return 0;
}

static void benchmarkConv2D() {

    std::array<int, 10> batch_sizes = { 1, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000 };
    for (int i = 0; i < batch_sizes.size(); ++i) {

        int batch_size = batch_sizes[i];

        Tensor X(batch_size, 1, 1, 28, 28); // NCDWH
        X.setRandom();

        // D_out = floor((D_in + 2 × padding_d - dilation_d × (kernel_d - 1) - 1) / stride_d + 1)
        // H_out = floor((H_in + 2 × padding_h - dilation_h × (kernel_h - 1) - 1) / stride_h + 1)
        // W_out = floor((W_in + 2 × padding_w - dilation_w × (kernel_w - 1) - 1) / stride_w + 1)
        ConvLayer conv1(1, 2, 3, 3, 1, 1, 1, 1);
        Tensor output;

        // Benchmark convolution 2D
        auto conv2d_times = BenchmarkTimer::measure([&]() {
            output = conv1.forward(X);
            (void)output;
            }, /* warmup= */ 3, /* reps= */ 10);

        auto conv2d_stats = BenchmarkTimer::compute_stats(conv2d_times, batch_size);

        // BenchmarkTimer::print_stats("CNN 2D - convolution", conv2d_stats);
        BenchmarkTimer::save_csv(RESULT_FILE, "exp0_gemm_conv2d_data_parallel", 1, conv2d_stats);
    }

}

int main(int argc, char* argv[]) {

    // // This should be done once.
    // std::ofstream baseline_file(RESULT_FILE, std::ios::out | std::ios::trunc);
    // baseline_file << "experiment, n_threads, means_ms, median_ms, std_ms,"
    //     "min_ms, max_ms, throughput_img_s, speedup, efficiency_pct\n";

    benchmarkConv2D();

    return 0;
}


