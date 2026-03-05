# include <chrono>
# include <iostream>
#include "Tensor.hpp"


class Timer {

private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;   
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
    }

    double elapsedMilliseconds() const {
        std::chrono::duration<double, std::milli> duration = end_time - start_time;
        return duration.count();
    }

};

class CNNBenchMarck {

public:
    static void runBenchmark(CNN & model,  const Tensor& test_input) {
        Timer timer;
        
        timer.start();
        model.forward(test_input);
        timer.stop();
        std::cout << "Forward pass time: " << timer.elapsedMilliseconds() << " ms" << std::endl;

        // Assuming we have a gradient tensor of the same shape as output
        Tensor gradOutput = test_input; // Placeholder for actual gradient tensor
        timer.start();
        model.backward(gradOutput);
        timer.stop();
        std::cout << "Backward pass time: " << timer.elapsedMilliseconds() << " ms" << std::endl;
    }
    
}