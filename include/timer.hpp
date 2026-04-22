// common/timer.hpp

#pragma once

#include <chrono>

#include <vector>

#include <numeric>

#include <algorithm>

#include <cmath>

#include <iostream>

#include <fstream>

#include <string>



class BenchmarkTimer {

public:

    using Clock = std::chrono::high_resolution_clock;

    using Ms = std::chrono::duration<double, std::milli>;



    // Lance N répétitions de fn(), retourne les temps en ms

    template<typename Fn>
    static std::vector<double> measure(Fn&& fn, int warmup = 3, int reps = 10) {

        // Warmup : remplir les caches

        for (int i = 0; i < warmup; ++i) fn();



        std::vector<double> times;

        times.reserve(reps);

        for (int i = 0; i < reps; ++i) {

            auto t0 = Clock::now();

            fn();

            auto t1 = Clock::now();

            times.push_back(Ms(t1 - t0).count());

        }

        return times;

    }



    struct Stats {
        double mean, median, std_dev, min, max;
        double batch_size; // si fourni à compute_stats
        double throughput_img_per_sec; // si batch_size fourni
    };


    static Stats compute_stats(const std::vector<double>& times, int batch_size = 1) {

        Stats s;

        auto sorted = times;

        std::sort(sorted.begin(), sorted.end());

        s.min = sorted.front();

        s.max = sorted.back();

        s.median = sorted[sorted.size() / 2];

        s.mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

        s.batch_size = batch_size;

        double sq_sum = 0;

        for (auto t : times) sq_sum += (t - s.mean) * (t - s.mean);

        s.std_dev = std::sqrt(sq_sum / times.size());

        s.throughput_img_per_sec = (batch_size / s.mean) * 1000.0;

        return s;

    }

    static void print_stats(const std::string& label, const Stats& s, const Stats* baseline = nullptr) {

        std::cout << "\n=== " << label << " ===\n";

        std::cout << " Latence moyenne : " << s.mean << " ms (±" << s.std_dev << ")\n";

        std::cout << " Médiane : " << s.median << " ms\n";

        std::cout << " Min / Max : " << s.min << " / " << s.max << " ms\n";

        std::cout << " Débit : " << s.throughput_img_per_sec << " img/s\n";

        if (baseline) {

            double speedup = baseline->mean / s.mean;

            std::cout << " Speedup : " << speedup << "x\n";

        }

    }

    static void save_csv(const std::string& filename, const std::string& exp_name,
    
        int n_threads, const Stats& s, const Stats* baseline = nullptr) {

        std::ofstream f(filename, std::ios::app);
        if(!f.is_open()){
            std::cerr << "Erreur : Impossible d'ouvrir le fichier ";
            return ;
        }
        double speedup = baseline ? baseline->mean / s.mean : 1.0;

        double efficiency = baseline ? speedup / n_threads * 100.0 : 100.0;
        f << exp_name << "," << n_threads << ","

            << s.mean << "," << s.median << "," << s.std_dev << ","

            << s.min << "," << s.max << ","
            << s.batch_size << ","
            << s.throughput_img_per_sec << "," << speedup << "," << efficiency 
            << std::endl;

    }

};




