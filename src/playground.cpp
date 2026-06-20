// #include "shared.hpp"
#include "CNNLIB.hpp"
#include "timer.hpp"
#include <iostream>
using namespace std;

#define BATCH_SIZE  15
#define Dout 256

void benchmarkDenseSequential() {
    int batch_size = BATCH_SIZE;
    int n = 28, c = 3;
    Tensor input(batch_size, c, n, n, n);
    input.setRandom();
    DenseLayer denselayer(n * n * n * c, Dout); // dimension d'entrée 64, dimension de sortie 128

    auto times_baseline = BenchmarkTimer::measure([&]() {
        Tensor out = denselayer.forward(input);
        Tensor grad = Tensor(out);
        grad.setRandom(); // gradient de sortie aléatoire
        denselayer.backward(grad);
        (void)out;
        }, 1, 2);

    auto stats_baseline = BenchmarkTimer::compute_stats(times_baseline, batch_size);
    BenchmarkTimer::print_stats("CNN2 DenseBaseline", stats_baseline);
    // BenchmarkTimer::save_csv("./logs/parallelisation/dense_bench.csv", "DenseBaseline" , 1, stats_baseline);
}

void benchmarkDenseDataParallel(int n_threads) {

    int batch_size = BATCH_SIZE;
    int n = 28, c = 3;
    Tensor input(batch_size, c, n, n, n); // batch de 16, dimension d'entrée 64
    DenseLayerDataParallel layer(n * n * n * c, Dout, n_threads); // dimension d'entrée 64, dimension de sortie 128
    input.setRandom(); // remplir d'inputs aléatoires
    auto times = BenchmarkTimer::measure([&]() {
        Tensor out = layer.forward(input);
        Tensor grad = Tensor(out);
        grad.setRandom(); // gradient de sortie aléatoire
        layer.backward(grad);
        (void)out;
        }, 1, 2);

    auto stats = BenchmarkTimer::compute_stats(times, batch_size);

    BenchmarkTimer::print_stats("CNN2 DenseDataParallel", stats);
    // BenchmarkTimer::save_csv("./logs/parallelisation/dense_data_parallel_bench.csv", stats);
}

void benchmarkDenseModelParallel(int n_threads) {

    int batch_size = BATCH_SIZE;
    int n = 28, c = 3;
    Tensor input(batch_size, c, n, n, n);
    DenseLayerModelParallel layer(n * n * n * c, Dout, n_threads);
    input.setRandom();

    auto times = BenchmarkTimer::measure([&]() {
        Tensor out = layer.forward(input);
        Tensor grad = Tensor(out);
        grad.setRandom();
        layer.backward(grad);
        (void)out;
        }, 1, 2);

    auto stats = BenchmarkTimer::compute_stats(times, batch_size);
    BenchmarkTimer::print_stats("CNN2 DenseModelParallel", stats);
    // BenchmarkTimer::save_csv("./logs/parallelisation/dense_model_parallel_bench.csv", stats);
}

int break_main(int argc, char* argv[]) {

    std::vector<int> n_threads = { 2, 4, 8 };

    cout << "\n=== Sequential ===" << endl;
    benchmarkDenseSequential();

    for (int i = 0; i < 3; i++) {
        cout << "\n=== running on (" << n_threads[i] << ") threads ===" << endl;

        cout << "\n=== Data Parallelism ===" << endl;
        benchmarkDenseDataParallel(n_threads[i]);

        cout << "\n=== Model Parallelism ===" << endl;
        benchmarkDenseModelParallel(n_threads[i]);
    }

    return 0;
}


int main(int argc, char* argv[]){

    Tensor X(100, 3, 12, 28, 28);
    X.setRandom();

    Tensor Y(100, 5, 1, 1, 1);
    Y.setRandom();

    Tensor W(100, 3, 12, 12, 12);
    W.setRandom();


    return 0;
}



