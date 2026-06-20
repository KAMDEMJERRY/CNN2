// DatasetManager.hpp
#pragma once
#include <string>
#include <vector>
#include <stdexcept>

// Chemins des datasets
#define FRACTURE_PATH "../dataset/fracturemnist3d"
#define NODULE_PATH "../dataset/nodulemnist3d"
#define VESSEL_PATH "../dataset/vesselmnist3d"
#define ADRENAL_PATH "../dataset/adrenalmnist3d"

// Chemins datasets 2D
#define MNIST_TEST_PATH    "../dataset/mnist_img/trainingSet/trainingSet/" 
#define MNIST_TRAIN_PATH    "../dataset/mnist_img/trainingSample/trainingSample/"
#define BLOODCELLS_TRAIN_PATH "../dataset/bloodcell/images/TRAIN/"
#define BLOODCELLS_TEST_PATH  "../dataset/bloodcell/images/TEST/"

class DatasetManager {
public:
    // Types de datasets 3D
    enum DatasetType3D {
        FRACTURE_3D,
        NODULE_3D,
        VESSEL_3D,
        ADRENAL_3D
    };
    
    // Types de datasets 2D
    enum DatasetType2D {
        MNIST_2D,
        BLOODCELLS_2D
    };
    
    // Structure d'information commune pour tous les datasets
    struct Info {
        std::string name;
        std::string path;
        std::vector<std::string> class_names;
        int num_classes;
        int num_channels;      // 1 pour grayscale, 3 pour RGB
        int dims;              // 2 ou 3
        int img_height;        // pour 2D
        int img_width;         // pour 2D
        int vol_size;          // pour 3D (cube)
        
        // Constructeur pour 3D
        Info(const std::string& n, const std::string& p, 
             const std::vector<std::string>& classes, int n_classes, 
             int channels, int dim, int vol_sz = 28)
            : name(n), path(p), class_names(classes), num_classes(n_classes),
              num_channels(channels), dims(dim), img_height(vol_sz), 
              img_width(vol_sz), vol_size(vol_sz) {}
        
        // Constructeur pour 2D
        Info(const std::string& n, const std::string& p,
             const std::vector<std::string>& classes, int n_classes,
             int channels, int dim, int height, int width)
            : name(n), path(p), class_names(classes), num_classes(n_classes),
              num_channels(channels), dims(dim), img_height(height),
              img_width(width), vol_size(0) {}
    };
    
    // ============== Gestion des datasets 3D ==============
    static Info getInfo(DatasetType3D type) {
        switch(type) {
            case FRACTURE_3D:
                return Info("FractureMNIST3D", FRACTURE_PATH,
                           {"No fracture", "Fracture T1", "Fracture T2"},
                           3, 1, 3, 28);
                
            case NODULE_3D:
                return Info("NoduleMNIST3D", NODULE_PATH,
                           {"Benign", "Malignant"},
                           2, 1, 3, 28);
                
            case VESSEL_3D:
                return Info("VesselMNIST3D", VESSEL_PATH,
                           {"No vessel", "Vessel"},
                           2, 1, 3, 28);
                
            case ADRENAL_3D:
                return Info("AdrenalMNIST3D", ADRENAL_PATH,
                           {"Normal", "Adenoma", "Metastasis", "Pheochromocytoma"},
                           4, 1, 3, 28);
                
            default:
                throw std::runtime_error("Unknown 3D dataset type");
        }
    }
    
    // ============== Gestion des datasets 2D ==============
    static Info getInfo(DatasetType2D type) {
        switch(type) {
            case MNIST_2D:
                return Info("MNIST", "",
                           {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"},
                           10, 1, 2, 28, 28);
                
            case BLOODCELLS_2D:
                return Info("BloodCells", "",
                           {"Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"},
                           4, 3, 2, 224, 224);  // RGB, 224x224 typical size
                
            default:
                throw std::runtime_error("Unknown 2D dataset type");
        }
    }
    
    // ============== Méthodes utilitaires ==============
    static std::vector<DatasetType3D> getAllDatasets3D() {
        return {FRACTURE_3D, NODULE_3D, VESSEL_3D, ADRENAL_3D};
    }
    
    static std::vector<DatasetType2D> getAllDatasets2D() {
        return {MNIST_2D, BLOODCELLS_2D};
    }
    
    static std::string getDatasetName(DatasetType3D type) {
        switch(type) {
            case FRACTURE_3D: return "FractureMNIST3D";
            case NODULE_3D:   return "NoduleMNIST3D";
            case VESSEL_3D:   return "VesselMNIST3D";
            case ADRENAL_3D:  return "AdrenalMNIST3D";
            default: return "Unknown";
        }
    }
    
    static std::string getDatasetName(DatasetType2D type) {
        switch(type) {
            case MNIST_2D:       return "MNIST";
            case BLOODCELLS_2D:  return "BloodCells";
            default: return "Unknown";
        }
    }
    
    // Obtenir les chemins spécifiques pour 2D (train/test séparés)
    static std::pair<std::string, std::string> getPaths2D(DatasetType2D type) {
        switch(type) {
            case MNIST_2D:
                return {MNIST_TRAIN_PATH, MNIST_TEST_PATH};
            case BLOODCELLS_2D:
                return {BLOODCELLS_TRAIN_PATH, BLOODCELLS_TEST_PATH};
            default:
                throw std::runtime_error("Unknown 2D dataset type");
        }
    }
};
