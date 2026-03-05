#include <iostream>
#include <chrono>
#include <random>

#include "ConvLayer.hpp"

using namespace std;

// Fonction pour générer un Tensor aléatoire pour les tests
Tensor generate_random_tensor(int batch_size, int channels, int height, int width) {
    Tensor tensor(batch_size, channels, height, width);
    
    // Initialisation du générateur aléatoire
    static default_random_engine generator;
    static normal_distribution<float> distribution(0.0f, 1.0f);
    
    // Remplir le tensor avec des valeurs aléatoires
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    tensor(b, c, h, w) = distribution(generator);
                }
            }
        }
    }
    
    return tensor;
}

// Fonction pour évaluer la performance de ConvLayer avec im2col
void evaluate_convolution_forward_im2col(int input_height, int input_width, int input_channels, 
                                         int num_filters, int filter_height, int filter_width,
                                         int padding_h, int padding_w, int stride_h, int stride_w, 
                                         int batch_size, int num_iterations = 10, int warmup_iterations = 3) {
    
    cout << "\n=== Évaluation des performances de ConvLayer::forward (im2col) ===\n";
    cout << "Configuration:\n";
    cout << "  Taille d'entrée: " << input_height << "x" << input_width << "\n";
    cout << "  Canaux d'entrée: " << input_channels << "\n";
    cout << "  Nombre de filtres: " << num_filters << "\n";
    cout << "  Taille des filtres: " << filter_height << "x" << filter_width << "\n";
    cout << "  Padding: " << padding_h << "x" << padding_w << "\n";
    cout << "  Stride: " << stride_h << "x" << stride_w << "\n";
    cout << "  Batch size: " << batch_size << "\n";
    cout << "  Nombre d'itérations: " << num_iterations << "\n";
    
    // Création de la couche de convolution avec im2col
    ConvLayer conv_layer(input_channels, num_filters, 
                        filter_height, filter_width,
                        stride_h, stride_w,
                        padding_h, padding_w);
    
    // Génération des données d'entrée
    Tensor input_tensor = generate_random_tensor(batch_size, input_channels, input_height, input_width);
    
    // Calcul des dimensions de sortie
    int output_height = (input_height - filter_height + 2 * padding_h) / stride_h + 1;
    int output_width = (input_width - filter_width + 2 * padding_w) / stride_w + 1;
    
    // Étape de warmup
    cout << "\nÉtape de warmup (" << warmup_iterations << " itérations)...\n";
    Tensor warmup_output;
    for (int i = 0; i < warmup_iterations; ++i) {
        warmup_output = conv_layer.forward(input_tensor);
    }
    
    // Mesure des performances
    vector<chrono::duration<double, milli>> execution_times;
    vector<double> im2col_times;
    vector<double> matmul_times;
    vector<double> reshape_times;
    
    cout << "\nDébut des mesures de performance...\n";
    for (int i = 0; i < num_iterations; ++i) {
        auto start_total = chrono::high_resolution_clock::now();
        
        Tensor output = conv_layer.forward(input_tensor);
        
        auto end_total = chrono::high_resolution_clock::now();
        auto duration_total = chrono::duration_cast<chrono::duration<double, milli>>(end_total - start_total);
        execution_times.push_back(duration_total);
        
        cout << "  Itération " << i + 1 << ": " << duration_total.count() << " ms\n";
    }
    
    // Calcul des statistiques
    double total_time = 0.0;
    double min_time = numeric_limits<double>::max();
    double max_time = 0.0;
    
    for (const auto& time : execution_times) {
        double ms = time.count();
        total_time += ms;
        min_time = min(min_time, ms);
        max_time = max(max_time, ms);
    }
    
    double avg_time = total_time / num_iterations;
    
    // Calcul de l'écart-type
    double variance = 0.0;
    for (const auto& time : execution_times) {
        double ms = time.count();
        variance += pow(ms - avg_time, 2);
    }
    double std_dev = sqrt(variance / num_iterations);
    
    // Affichage des résultats
    cout << "\n=== Résultats des performances ===\n";
    cout << "Temps moyen de forward: " << avg_time << " ms\n";
    cout << "Temps minimum: " << min_time << " ms\n";
    cout << "Temps maximum: " << max_time << " ms\n";
    cout << "Écart-type: " << std_dev << " ms\n";
    cout << "Temps total: " << total_time << " ms\n";
    
    // Calcul des opérations pour im2col
    long long total_ops = static_cast<long long>(batch_size) * num_filters * input_channels *
                         output_height * output_width * filter_height * filter_width * 2;
    
    // Calcul de la taille de la matrice im2col
    int patch_size = input_channels * filter_height * filter_width;
    int num_patches = batch_size * output_height * output_width;
    long long im2col_memory = patch_size * num_patches * sizeof(float);
    
    cout << "\n=== Analyse des opérations (im2col) ===\n";
    cout << "Taille de sortie: " << output_height << "x" << output_width << "\n";
    cout << "Taille des patches: " << patch_size << "\n";
    cout << "Nombre de patches: " << num_patches << "\n";
    cout << "Mémoire im2col: " << im2col_memory / 1024.0 / 1024.0 << " MB\n";
    cout << "Nombre total d'opérations (mult-adds): " << total_ops << "\n";
    cout << "Giga-OPs: " << total_ops / 1e9 << "\n";
    cout << "Performance: " << (total_ops / (avg_time * 1e6)) << " GOPs/s\n";
}

// Fonction pour comparer les deux implémentations avec la même configuration
void compare_implementations_same_config() {
    cout << "\n=== Comparaison des deux implémentations ===\n";
    
    // Configuration commune
    int input_size = 32;
    int input_channels = 3;
    int num_filters = 64;
    int filter_size = 3;
    int padding = 1;
    int stride = 1;
    int batch_size = 16;
    int num_iterations = 5;
    
    // Configuration im2col (utilise les mêmes paramètres)
    int filter_height = filter_size;
    int filter_width = filter_size;
    int padding_h = padding;
    int padding_w = padding;
    int stride_h = stride;
    int stride_w = stride;
    
    cout << "\n--- Configuration de test ---\n";
    cout << "Taille d'entrée: " << input_size << "x" << input_size << "\n";
    cout << "Canaux d'entrée: " << input_channels << "\n";
    cout << "Nombre de filtres: " << num_filters << "\n";
    cout << "Taille des filtres: " << filter_size << "x" << filter_size << "\n";
    cout << "Padding: " << padding << "\n";
    cout << "Stride: " << stride << "\n";
    cout << "Batch size: " << batch_size << "\n";
    
    // Test de l'implémentation im2col
    cout << "\n=== Test de ConvLayer (im2col) ===";
    evaluate_convolution_forward_im2col(
        input_size, input_size,    // input_height, input_width
        input_channels,            // input_channels
        num_filters,               // num_filters
        filter_height, filter_width, // filter dimensions
        padding_h, padding_w,      // padding
        stride_h, stride_w,        // stride
        batch_size,                // batch_size
        num_iterations,            // iterations
        2                          // warmup
    );
}

// Fonction pour tester différentes configurations
void benchmark_different_configurations_im2col() {
    cout << "=== Benchmark de différentes configurations (im2col) ===\n";
    
    // Configuration 1: Petite taille (MNIST style)
    cout << "\n--- Configuration 1 (Petite) ---";
    evaluate_convolution_forward_im2col(
        28, 28,    // input_height, input_width
        1,         // input_channels (grayscale)
        32,        // num_filters
        3, 3,      // filter_height, filter_width
        1, 1,      // padding_h, padding_w
        1, 1,      // stride_h, stride_w
        32,        // batch_size
        10,        // iterations
        3          // warmup
    );
    
    // Configuration 2: Taille moyenne (CIFAR style)
    cout << "\n\n--- Configuration 2 (Moyenne) ---";
    evaluate_convolution_forward_im2col(
        64, 64,    // input_height, input_width
        3,         // input_channels (RGB)
        64,        // num_filters
        5, 5,      // filter_height, filter_width
        2, 2,      // padding_h, padding_w
        2, 2,      // stride_h, stride_w
        16,        // batch_size
        10,        // iterations
        3          // warmup
    );
    
    // Configuration 3: Grande taille (ImageNet style)
    cout << "\n\n--- Configuration 3 (Grande) ---";
    evaluate_convolution_forward_im2col(
        128, 128,  // input_height, input_width
        64,        // input_channels
        128,       // num_filters
        3, 3,      // filter_height, filter_width
        1, 1,      // padding_h, padding_w
        1, 1,      // stride_h, stride_w
        8,         // batch_size
        10,        // iterations
        3          // warmup
    );
}

// Fonction pour analyser l'impact du batch size
void analyze_batch_size_impact_im2col() {
    int input_height = 32;
    int input_width = 32;
    int input_channels = 3;
    int num_filters = 64;
    int filter_height = 3;
    int filter_width = 3;
    int padding_h = 1;
    int padding_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    
    cout << "\n=== Analyse de l'impact du batch size (im2col) ===\n";
    cout << "Configuration fixe:\n";
    cout << "  Taille d'entrée: " << input_height << "x" << input_width << "\n";
    cout << "  Canaux d'entrée: " << input_channels << "\n";
    cout << "  Nombre de filtres: " << num_filters << "\n";
    cout << "  Taille des filtres: " << filter_height << "x" << filter_width << "\n";
    
    vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64};
    
    for (int batch_size : batch_sizes) {
        cout << "\n--- Batch size: " << batch_size << " ---\n";
        
        ConvLayer conv_layer(input_channels, num_filters, 
                           filter_height, filter_width,
                           stride_h, stride_w,
                           padding_h, padding_w);
        
        Tensor input = generate_random_tensor(batch_size, input_channels, input_height, input_width);
        
        // Warmup
        for (int i = 0; i < 3; ++i) {
            conv_layer.forward(input);
        }
        
        // Mesure
        auto start = chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            conv_layer.forward(input);
        }
        auto end = chrono::high_resolution_clock::now();
        
        auto duration = chrono::duration_cast<chrono::duration<double, milli>>(end - start);
        double avg_time = duration.count() / 10.0;
        
        // Calcul des dimensions de sortie
        int output_height = (input_height - filter_height + 2 * padding_h) / stride_h + 1;
        int output_width = (input_width - filter_width + 2 * padding_w) / stride_w + 1;
        
        // Calcul de la mémoire im2col
        int patch_size = input_channels * filter_height * filter_width;
        int num_patches = batch_size * output_height * output_width;
        double im2col_memory_mb = (patch_size * num_patches * sizeof(float)) / 1024.0 / 1024.0;
        
        cout << "Temps moyen par forward: " << avg_time << " ms\n";
        cout << "Temps par échantillon: " << avg_time / batch_size << " ms\n";
        cout << "Mémoire im2col: " << im2col_memory_mb << " MB\n";
    }
}

// Fonction pour analyser l'efficacité de l'utilisation mémoire
void analyze_memory_efficiency() {
    cout << "\n=== Analyse de l'efficacité mémoire (im2col) ===\n";
    
    vector<int> input_sizes = {16, 32, 64, 128};
    int input_channels = 3;
    int num_filters = 64;
    int filter_size = 3;
    int padding = 1;
    int stride = 1;
    int batch_size = 16;
    
    cout << "Paramètres fixes:\n";
    cout << "  Canaux d'entrée: " << input_channels << "\n";
    cout << "  Nombre de filtres: " << num_filters << "\n";
    cout << "  Taille des filtres: " << filter_size << "x" << filter_size << "\n";
    cout << "  Padding: " << padding << "\n";
    cout << "  Stride: " << stride << "\n";
    cout << "  Batch size: " << batch_size << "\n\n";
    
    for (int input_size : input_sizes) {
        cout << "--- Taille d'entrée: " << input_size << "x" << input_size << " ---\n";
        
        // Calcul de la mémoire
        int output_size = (input_size - filter_size + 2 * padding) / stride + 1;
        int patch_size = input_channels * filter_size * filter_size;
        int num_patches = batch_size * output_size * output_size;
        
        double input_memory = (batch_size * input_channels * input_size * input_size * sizeof(float)) / 1024.0 / 1024.0;
        double output_memory = (batch_size * num_filters * output_size * output_size * sizeof(float)) / 1024.0 / 1024.0;
        double im2col_memory = (patch_size * num_patches * sizeof(float)) / 1024.0 / 1024.0;
        double weights_memory = (num_filters * input_channels * filter_size * filter_size * sizeof(float)) / 1024.0;
        
        cout << "  Mémoire d'entrée: " << input_memory << " MB\n";
        cout << "  Mémoire de sortie: " << output_memory << " MB\n";
        cout << "  Mémoire im2col: " << im2col_memory << " MB\n";
        cout << "  Mémoire des poids: " << weights_memory << " KB\n";
        cout << "  Mémoire totale (im2col): " << input_memory + output_memory + im2col_memory << " MB\n";
        cout << "  Facteur d'expansion im2col: " << (im2col_memory / input_memory) << "x\n";
    }
}

int main() {
    cout << "Programme d'évaluation des performances de ConvLayer::forward (im2col)\n";
    
    // Menu pour choisir le test
    int choice;
    do {
        cout << "\nMenu:\n";
        cout << "1. Tester une configuration spécifique (im2col)\n";
        cout << "2. Benchmark de différentes configurations (im2col)\n";
        cout << "3. Analyser l'impact du batch size (im2col)\n";
        cout << "4. Comparer avec l'autre implémentation (configuration commune)\n";
        cout << "5. Analyser l'efficacité mémoire\n";
        cout << "6. Quitter\n";
        cout << "Choix: ";
        cin >> choice;
        
        switch (choice) {
            case 1: {
                int input_h, input_w, input_channels, num_filters;
                int filter_h, filter_w, padding_h, padding_w, stride_h, stride_w, batch_size;
                
                cout << "\nEntrez les paramètres:\n";
                cout << "Hauteur d'entrée: ";
                cin >> input_h;
                cout << "Largeur d'entrée: ";
                cin >> input_w;
                cout << "Nombre de canaux d'entrée: ";
                cin >> input_channels;
                cout << "Nombre de filtres: ";
                cin >> num_filters;
                cout << "Hauteur des filtres: ";
                cin >> filter_h;
                cout << "Largeur des filtres: ";
                cin >> filter_w;
                cout << "Padding (hauteur): ";
                cin >> padding_h;
                cout << "Padding (largeur): ";
                cin >> padding_w;
                cout << "Stride (hauteur): ";
                cin >> stride_h;
                cout << "Stride (largeur): ";
                cin >> stride_w;
                cout << "Batch size: ";
                cin >> batch_size;
                
                evaluate_convolution_forward_im2col(input_h, input_w, input_channels, num_filters,
                                                  filter_h, filter_w, padding_h, padding_w,
                                                  stride_h, stride_w, batch_size);
                break;
            }
            case 2:
                benchmark_different_configurations_im2col();
                break;
            case 3:
                analyze_batch_size_impact_im2col();
                break;
            case 4:
                compare_implementations_same_config();
                break;
            case 5:
                analyze_memory_efficiency();
                break;
            case 6:
                cout << "Au revoir!\n";
                break;
            default:
                cout << "Choix invalide!\n";
        }
    } while (choice != 6);
    
    return 0;
}