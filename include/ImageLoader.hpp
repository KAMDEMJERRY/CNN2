# pragma once


# include <opencv2/opencv.hpp>
# include <filesystem>
# include <vector>
# include <string>
# include <random>
# include <algorithm>
#include <fstream>    
#include <iomanip> 
# include "Tensor.hpp"


namespace fs = std::filesystem;

class ImageLoader {
private:
    int target_width;
    int target_height;
    bool normalize;
    int num_channels; // 1 pour Gray, 3 pour RGB

    float rgb_mean[3] = { 0.485f, 0.456f, 0.406f };
    float rgb_std[3] = { 0.229f, 0.224f, 0.225f };
    float gray_mean = 0.5f;
    float gray_std = 0.5f;


    // Statistiques calculées
    bool stats_computed = false;
    float computed_mean[3] = { 0.0f };
    float computed_std[3] = { 0.0f };

public:
    // Constructeur pour RGB
    ImageLoader(int width = 224, int height = 224, bool normalize = true, int channels = 3) :
        target_width(width), target_height(height), normalize(normalize), num_channels(channels) {
    }

    // Constructeur avec statistiques personnalisées
    ImageLoader(int width, int height, bool normalize, int channels,
        const float* custom_mean, const float* custom_std)
        : target_width(width), target_height(height), normalize(normalize), num_channels(channels) {

        if (custom_mean && custom_std) {
            if (channels == 1) {
                gray_mean = custom_mean[0];
                gray_std = custom_std[0];
            }
            else {
                for (int i = 0; i < 3; ++i) {
                    rgb_mean[i] = custom_mean[i];
                    rgb_std[i] = custom_std[i];
                }
            }
            stats_computed = true;
        }
    }
    int getTargetWidth() const { return target_width; }   // const
    int getTargetHeight() const { return target_height; } // const
    int getNumChannels() const { return num_channels; }   // const
    bool isNormalized() const { return normalize; }       // const

    // Pour les stats
    float getMean(int channel = 0) const { return computed_mean[channel]; }
    float getStd(int channel = 0) const { return computed_std[channel]; }
    bool hasStatistics() const { return stats_computed; }

    // MÉTHODE PRINCIPALE : Calcul automatique des stats
    void computeStatistics(const std::vector<std::string>& image_paths, bool verbose = true) {
        if (image_paths.empty()) {
            throw std::runtime_error("No images to compute statistics");
        }

        std::cout << "\n=== 📊 Calcul automatique des statistiques ===" << std::endl;
        std::cout << "Images: " << image_paths.size() << std::endl;
        std::cout << "Taille: " << target_width << "x" << target_height << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Initialisation des accumulateurs
        double sum[3] = { 0.0, 0.0, 0.0 };
        double sum_sq[3] = { 0.0, 0.0, 0.0 };
        size_t total_pixels = 0;
        int valid_images = 0;

        // Déterminer le type d'image (première image)
        cv::Mat first_img = cv::imread(image_paths[0], cv::IMREAD_UNCHANGED);
        int channels = first_img.channels();
        int output_channels = (channels == 1) ? 1 : 3;

#pragma omp parallel for reduction(+:sum,sum_sq,total_pixels,valid_images)
        for (size_t i = 0; i < image_paths.size(); ++i) {
            try {
                cv::Mat img = cv::imread(image_paths[i], cv::IMREAD_UNCHANGED);
                if (img.empty()) continue;

                // Redimensionner
                cv::Mat resized;
                cv::resize(img, resized, cv::Size(target_width, target_height));

                // Convertir en float [0,1]
                if (output_channels == 1) {
                    cv::Mat float_img;
                    if (img.channels() > 1) {
                        cv::cvtColor(resized, resized, cv::COLOR_BGR2GRAY);
                    }
                    resized.convertTo(float_img, CV_32FC1, 1.0 / 255.0);

                    for (int h = 0; h < target_height; ++h) {
                        for (int w = 0; w < target_width; ++w) {
                            float val = float_img.at<float>(h, w);
                            sum[0] += val;
                            sum_sq[0] += val * val;
                            total_pixels++;
                        }
                    }
                }
                else {
                    cv::Mat rgb;
                    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
                    cv::Mat float_img;
                    rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

                    std::vector<cv::Mat> ch(3);
                    cv::split(float_img, ch);

                    for (int c = 0; c < 3; ++c) {
                        for (int h = 0; h < target_height; ++h) {
                            for (int w = 0; w < target_width; ++w) {
                                float val = ch[c].at<float>(h, w);
                                sum[c] += val;
                                sum_sq[c] += val * val;
                            }
                        }
                    }
                    total_pixels += target_height * target_width * 3;
                }
                valid_images++;

            }
            catch (...) {
                // Ignorer les erreurs
            }
        }

        // Calcul des statistiques finales
        double pixel_count = total_pixels / output_channels;

        for (int c = 0; c < output_channels; ++c) {
            computed_mean[c] = sum[c] / pixel_count;
            computed_std[c] = std::sqrt((sum_sq[c] / pixel_count) - (computed_mean[c] * computed_mean[c]));
        }

        // Mettre à jour les statistiques utilisées pour la normalisation
        if (output_channels == 1) {
            gray_mean = computed_mean[0];
            gray_std = computed_std[0];
            num_channels = 1;
        }
        else {
            for (int i = 0; i < 3; ++i) {
                rgb_mean[i] = computed_mean[i];
                rgb_std[i] = computed_std[i];
            }
            num_channels = 3;
        }

        stats_computed = true;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

        if (verbose) {
            printStatistics(output_channels, duration.count());
        }
    }

    // Afficher les statistiques
    void printStatistics(int channels, long seconds) const {
        std::cout << "\n📊 Statistiques calculées en " << seconds << "s" << std::endl;
        std::cout << "═══════════════════════════════════════════════════" << std::endl;

        if (channels == 1) {
            std::cout << "🖤 Grayscale - 1 canal" << std::endl;
            std::cout << "   Mean: " << std::fixed << std::setprecision(6) << computed_mean[0] << std::endl;
            std::cout << "   Std:  " << std::setprecision(6) << computed_std[0] << std::endl;
        }
        else {
            std::cout << "🎨 RGB - 3 canaux" << std::endl;
            const char* channel_names[3] = { "R", "G", "B" };
            for (int i = 0; i < 3; ++i) {
                std::cout << "   Channel " << channel_names[i] << ": "
                    << "Mean = " << std::setprecision(6) << computed_mean[i]
                    << ", Std = " << std::setprecision(6) << computed_std[i] << std::endl;
            }
        }
        std::cout << "═══════════════════════════════════════════════════" << std::endl;

        // Format copiable
        std::cout << "\n📋 À copier dans votre code:" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        if (channels == 1) {
            std::cout << "float mean = " << std::setprecision(6) << computed_mean[0] << "f;" << std::endl;
            std::cout << "float std  = " << std::setprecision(6) << computed_std[0] << "f;" << std::endl;
        }
        else {
            std::cout << "float mean[3] = {"
                << std::setprecision(6) << computed_mean[0] << "f, "
                << computed_mean[1] << "f, "
                << computed_mean[2] << "f};" << std::endl;
            std::cout << "float std[3]  = {"
                << std::setprecision(6) << computed_std[0] << "f, "
                << computed_std[1] << "f, "
                << computed_std[2] << "f};" << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }

    // Detection auto du nombre de canaux
    int getImageChannels(const cv::Mat& img) {
        return img.channels();
    }



    Tensor preprocessImage(const cv::Mat& img) {
        // Detecter le type dimage
        int channels = img.channels();

        if (channels == 1) {
            return preprocessGrayImage(img);
        }
        else if (channels == 3) {
            return preprocessRGBImage(img);
        }
        else if (channels == 4) {
            // Convertir RGBA -> RGB
            cv::Mat rgb;
            cv::cvtColor(img, rgb, cv::COLOR_BGRA2BGR);
            return preprocessRGBImage(rgb);
        }
        else {
            throw std::runtime_error("Unsopported number of channels: " + std::to_string(channels));
        }
    }

    // Traitement pour les images niveaux de gris
    Tensor preprocessGrayImage(const cv::Mat& img) {
        // Redimensionner
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(target_width, target_height));

        // Convertir en float [0, 1]
        cv::Mat float_img;
        resized.convertTo(float_img, CV_32FC1, 1.0 / 255.0);

        // Normalisation
        if (normalize) {
            float_img = (float_img - gray_mean) / gray_std;
        }

        // cv::imshow("Ma Matrice", float_img);
        // cv::waitKey(0);
        // std::cout << "M = " << std::endl << " " << float_img << std::endl;

        // Creer Tensor [1, 1, H, W] (batch = 1, channel = 1)
        Tensor tensor(1, 1, target_height, target_width);

        for (int h = 0; h < target_height; h++) {
            for (int w = 0; w < target_width; w++) {
                tensor(0, 0, h, w) = float_img.at<float>(h, w);
            }
        }

        return tensor;
    }


    // Traitement pour les images RGB
    Tensor preprocessRGBImage(const cv::Mat& img) {
        // Convertir BGR -> RGB
        cv::Mat rgb;
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

        // Redimensionner
        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(target_width, target_height));

        // Convertir en float [0, 1]
        cv::Mat float_img;
        resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

        // Normalisation par canal
        if (normalize) {
            std::vector<cv::Mat> channels(3);
            cv::split(float_img, channels);

            for (int c = 0; c < 3; c++) {
                channels[c] = (channels[c] - rgb_mean[c]) / rgb_std[c];
            }

            cv::merge(channels, float_img);
        }

        // Creer Tensor [1, 3, H, W]
        Tensor tensor(1, 3, target_height, target_width);

        std::vector<cv::Mat> channels(3);
        cv::split(float_img, channels);

        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < target_height; h++) {
                for (int w = 0; w < target_width; w++) {
                    tensor(0, c, h, w) = channels[c].at<float>(h, w);
                }
            }
        }

        return tensor;
    }

    // Charger un batch avec un support mixte RGB/GRIS
    Tensor loadBatch(const std::vector<std::string>& filepaths) {
        int batch_size = filepaths.size();

        // Lire la premiere image pour determiner le nombre de canaux
        cv::Mat first_img = cv::imread(filepaths[0], cv::IMREAD_UNCHANGED);
        int channels = first_img.channels();
        int output_channels = (channels == 1) ? 1 : 3;

        // Creer le tensor avec le bon nombre de canaux
        Tensor batch(batch_size, output_channels, target_height, target_width);

# pragma omp parallel for
        for (int i = 0; i < batch_size; i++) {
            try {
                cv::Mat img = cv::imread(filepaths[i], cv::IMREAD_UNCHANGED);
                Tensor img_tensor = preprocessImage(img);

                // Copier dans le batch
                int img_channels = img_tensor.dim(1);
                for (int c = 0; c < img_channels; c++) {
                    for (int h = 0; h < target_height; h++) {
                        for (int w = 0; w < target_width; w++) {
                            batch(i, c, h, w) = img_tensor(0, c, h, w);
                        }
                    }
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Error loading " << filepaths[i] << ": " << e.what() << std::endl;

                // Mettre des zeros
                for (int c = 0; c < output_channels; c++) {
                    for (int h = 0; h < target_height; h++) {
                        for (int w = 0; w < target_width; w++) {
                            batch(i, c, h, w) = 0.0f;
                        }
                    }
                }
            }
        }
        return batch;
    }

    // Post-traitement pour afficher GRaY
    cv::Mat tensorToGrayImage(const Tensor& tensor, int batch_idx = 0) {
        int height = tensor.dim(2);
        int width = tensor.dim(3);

        cv::Mat img(height, width, CV_32FC1);

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                float val = tensor(batch_idx, 0, h, w);
                if (normalize) {
                    val = val * gray_std + gray_mean;
                }
                img.at<float>(h, w) = val;
            }
        }

        // Convertir de float [0,1] à uint8 [0,255] pour affichage
        cv::Mat img_8u;
        img.convertTo(img_8u, CV_8UC1, 255.0);

        return img_8u;
    }

    cv::Mat tensorToImage(const Tensor& tensor, int batch_idx = 0) {
        int height = tensor.dim(2);
        int width = tensor.dim(3);

        std::vector<cv::Mat> channels(3);

        for (int c = 0; c < 3; c++) {
            channels[c] = cv::Mat(height, width, CV_32FC1);
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    float val = tensor(batch_idx, c, h, w);
                    if (normalize) {
                        val = val * rgb_std[c] + rgb_mean[c];
                    }
                    channels[c].at<float>(h, w) = val * 255.0f;
                }
            }
        }


        cv::Mat rgb;
        cv::merge(channels, rgb);

        // Convertir de float [0,1] à uint8 [0,255]
        cv::Mat rgb_8u;
        rgb.convertTo(rgb_8u, CV_8UC3, 255.0);


        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);

        return bgr;
    }

    void showTensorGrayImage(Tensor& images, int idx = 0) {
        cv::Mat mat = tensorToGrayImage(images, idx);  // Ajout du type cv::Mat et =
        cv::imshow("Image", mat);  // cv::imshow, pas mat.imshow()
        cv::waitKey(0);  // Nécessaire pour afficher
    }
    
    void showTensorRGBImage(Tensor& images, int idx = 0) {
        cv::Mat mat = tensorToImage(images, idx);  // Ajout du type cv::Mat et =
        cv::imshow("Image", mat);  // cv::imshow, pas mat.imshow()
        cv::waitKey(0);  // Nécessaire pour afficher
    }


};










class ImageFolderDataset {
private:
    struct Sample {
        std::string path;
        int label;
        std::string class_name;
        int channels; // le nombre de canaux
    };

    std::vector<Sample> samples;
    std::vector<std::string> class_names;
    ImageLoader loader;
    bool is_grayscale;

    // Statistiques du dataset
    bool stats_computed = false;
    float dataset_mean[3] = { 0.0f };
    float dataset_std[3] = { 0.0f };

public:

    ImageLoader& getImageLoader() {
        return loader;
    }

    ImageFolderDataset(const std::string& root_dir,
        int target_width = 224,
        int target_height = 224,
        bool normalize = true,
        bool force_grayscale = false,
        bool auto_compute_stats = true) :
        loader(target_width, target_height, false, force_grayscale ? 1 : 3) {
        loadFromFolder(root_dir);
        detectImageType();

        if (auto_compute_stats) {
            computeStatistics();
        }

        // Si normalize = true, on met à jour le loader avec les stats calculées
        if (normalize) {
            enableNormalization();
        }
    }

    // Calcul automatique des statistiques
    void computeStatistics(bool verbose = true) {
        std::cout << "\n=== 🔬 Calcul automatique des statistiques du dataset ===" << std::endl;

        // Extraire tous les chemins d'images
        std::vector<std::string> paths;
        for (const auto& sample : samples) {
            paths.push_back(sample.path);
        }

        // Calculer les stats via l'ImageLoader
        loader.computeStatistics(paths, verbose);

        // Récupérer les stats calculées
        int channels = getNumChannels();
        for (int i = 0; i < channels; ++i) {
            dataset_mean[i] = loader.getMean(i);
            dataset_std[i] = loader.getStd(i);
        }
        stats_computed = true;

        // Afficher un résumé
        std::cout << "\n📈 Statistiques du dataset:" << std::endl;
        std::cout << "   Images: " << samples.size() << std::endl;
        std::cout << "   Classes: " << class_names.size() << std::endl;
        std::cout << "   Canaux: " << channels << std::endl;
    }

    // Activer la normalisation avec les stats calculées
    void enableNormalization() {
        if (!stats_computed) {
            std::cerr << "⚠️  Statistiques non calculées. Calcul automatique..." << std::endl;
            computeStatistics(false);
        }

        // Créer un nouveau loader avec les stats calculées
        int channels = getNumChannels();
        if (channels == 1) {
            loader = ImageLoader(loader.getTargetWidth(),
                loader.getTargetHeight(),
                true, 1,
                &dataset_mean[0], &dataset_std[0]);
        }
        else {
            loader = ImageLoader(loader.getTargetWidth(),
                loader.getTargetHeight(),
                true, 3,
                dataset_mean, dataset_std);
        }

        std::cout << "✅ Normalisation activée avec les statistiques du dataset" << std::endl;
    }

    // Sauvegarder les stats dans un fichier
    void saveStatistics(const std::string& filename) const {
        if (!stats_computed) {
            throw std::runtime_error("Statistics not computed yet");
        }

        std::ofstream file(filename);
        file << std::setprecision(6) << std::fixed;
        file << "# Dataset statistics\n";
        file << "samples " << samples.size() << "\n";
        file << "classes " << class_names.size() << "\n";
        file << "channels " << getNumChannels() << "\n";
        file << "width " << loader.getTargetWidth() << "\n";
        file << "height " << loader.getTargetHeight() << "\n";

        int channels = getNumChannels();
        file << "mean";
        for (int i = 0; i < channels; ++i) file << " " << dataset_mean[i];
        file << "\nstd";
        for (int i = 0; i < channels; ++i) file << " " << dataset_std[i];
        file << "\n";

        std::cout << "💾 Statistiques sauvegardées dans " << filename << std::endl;
    }

    // Charger des stats depuis un fichier
    void loadStatistics(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open statistics file: " + filename);
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            std::string key;
            iss >> key;

            if (key == "mean") {
                int channels = getNumChannels();
                for (int i = 0; i < channels; ++i) iss >> dataset_mean[i];
            }
            else if (key == "std") {
                int channels = getNumChannels();
                for (int i = 0; i < channels; ++i) iss >> dataset_std[i];
            }
        }

        stats_computed = true;
        enableNormalization();
        std::cout << "📂 Statistiques chargées depuis " << filename << std::endl;
    }

    // Getters pour les stats
    float getDatasetMean(int channel = 0) const { return dataset_mean[channel]; }
    float getDatasetStd(int channel = 0) const { return dataset_std[channel]; }
    bool hasStatistics() const { return stats_computed; }

    void loadFromFolder(const std::string& root_dir) {
        samples.clear();
        class_names.clear();


        // Parcourir les sous-dossiers (chaque sous-dossier = une classe)
        int label = 0;
        for (const auto& entry : fs::directory_iterator(root_dir)) {
            if (entry.is_directory()) {
                std::string class_name = entry.path().filename().string();
                class_names.push_back(class_name);

                // Parcourir les images dans ce dossier
                for (const auto& img_entry : fs::recursive_directory_iterator(entry.path())) {

                    if (img_entry.is_regular_file()) {
                        std::string ext = img_entry.path().extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                            samples.push_back({
                                img_entry.path().string(),
                                label,
                                class_name
                                });
                        }
                    }
                }
                label++;
            }
        }

        std::cout << "Loaded " << samples.size() << " imges from "
            << class_names.size() << " classes" << std::endl;

        for (int i = 0; i < class_names.size(); i++) {
            std::cout << " Class " << i << ": " << class_names[i] << std::endl;
        }
    }

    int getNumClasses() const { return class_names.size(); }
    int getNumSamples() const { return samples.size(); }
    const std::vector<std::string>& getClassNames() const { return class_names; }


    void detectImageType() {
        is_grayscale = true;
        for (const auto& sample : samples) {
            cv::Mat img = cv::imread(sample.path, cv::IMREAD_UNCHANGED);
            if (img.channels() > 1) {
                is_grayscale = false;
                break;
            }
        }

        std::cout << "Dataset type: " << (is_grayscale ? "Grayscale" : "RGB") << std::endl;
    }

    int getNumChannels() const {
        return is_grayscale ? 1 : 3;
    }

    std::pair<Tensor, Tensor> getBatch(const std::vector<int>& indices) {
        std::vector<std::string> paths;
        std::vector<int> labels;

        for (int idx : indices) {
            paths.push_back(samples[idx].path);
            labels.push_back(samples[idx].label);
        }

        Tensor images = loader.loadBatch(paths);
        Tensor targets = labelsToOneHot(labels, getNumClasses());

        return { images, targets };
    }

    // Convertir labels en one_hot
    Tensor labelsToOneHot(const std::vector<int>& labels, int num_classes) {
        int batch_size = labels.size();
        Tensor targets(batch_size, num_classes, 1, 1);
        targets.setZero();
        // targets.printShape();

        for (int i = 0; i < batch_size; i++) {
            // std::cout << "Label[" << i << "] : " << labels[i] << std::endl;
            if (labels[i] >= 0 && labels[i] < num_classes) {
                targets(i, labels[i], 0, 0) = 1.0f;
            }

            // targets.printByChannel(targets.dim(0), targets.dim(1), targets.dim(2), targets.dim(3));

        }

        return targets;
    }

    // Convertir predictions en labels
    std::vector<int> predictionsToLabels(const Tensor& predictions) {
        int batch_size = predictions.dim(0);
        int num_classes = predictions.dim(1);
        std::vector<int> labels(batch_size);

        for (int b = 0; b < batch_size; b++) {
            int max_class = 0;
            float max_prob = predictions(b, 0, 0, 0);

            for (int c = 1; c < num_classes; c++) {
                if (predictions(b, c, 0, 0) > max_prob) {
                    max_prob = predictions(b, c, 0, 0);
                    max_class = c;
                }
            }
            labels[b] = max_class;
        }

        return labels;
    }

    // Calculer la precision
    float computeAccuracy(const Tensor& predictions, const Tensor& targets) {
        std::vector<int> pred_labels = predictionsToLabels(predictions);
        int batch_size = predictions.dim(0);
        int correct = 0;

        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < targets.dim(1); c++) {
                if (targets(b, c, 0, 0) > 0.5f) {
                    if (c == pred_labels[b]) {
                        correct++;
                    }
                    break;
                }
            }
        }

        return static_cast<float>(correct) / batch_size;
    }
};