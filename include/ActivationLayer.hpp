
# include "Layer.hpp"

class ReLULayer : public Layer {
private:
    Tensor input_cache;
    
public:
    ReLULayer() = default;
    ~ReLULayer() override = default;
    
    Tensor forward(const Tensor& input) override {
        input_cache = input;
        Tensor output = input;
        output.eigen() = output.eigen().cwiseMax(0.0f);
        return output;
    }
    
    Tensor backward(const Tensor& gradOutput) override {
        Tensor gradInput(gradOutput.shape());
        
        // ✅ SOLUTION: Version Eigen correcte
        auto& input_tensor = input_cache.eigen();
        auto& grad_tensor = gradOutput.eigen();
        auto& output_tensor = gradInput.eigen();
        
        // Créer un masque binaire (1 où input > 0, 0 ailleurs)
        Eigen::Tensor<float, 4, Eigen::RowMajor> mask = 
            (input_tensor > 0.0f).template cast<float>();
        
        // Multiplier le gradient par le masque
        output_tensor = grad_tensor * mask;
        
        return gradInput;
    }
    
    std::string getName() const override {
        return "ReLU";
    }
};
class LeakyReLULayer : public Layer {
private:
    Tensor input_cache;
    float alpha;
    
public:
    LeakyReLULayer(float alpha = 0.01f) : alpha(alpha) {}
    ~LeakyReLULayer() override = default;
    
    Tensor forward(const Tensor& input) override {
        input_cache = input;
        Tensor output = input;
        
        // Leaky ReLU: x si x > 0, alpha*x sinon
        for (int i = 0; i < input.size(); ++i) {
            output[i] = (input[i] > 0.0f) ? input[i] : alpha * input[i];
        }
        
        return output;
    }
    
    Tensor backward(const Tensor& gradOutput) override {
        if (gradOutput.shape() != input_cache.shape()) {
            throw std::runtime_error("LeakyReLU: shape mismatch in backward");
        }
        
        Tensor gradInput(gradOutput.shape());
        
        // Gradient: 1 si x > 0, alpha sinon
        for (int i = 0; i < gradOutput.size(); ++i) {
            gradInput[i] = (input_cache[i] > 0.0f) ? 
                           gradOutput[i] : 
                           alpha * gradOutput[i];
        }
        
        return gradInput;
    }
    
    std::string getName() const override {
        return "LeakyReLU";
    }
};
class SigmoidLayer : public Layer {
public:
    SigmoidLayer() = default;  
    ~SigmoidLayer() override = default; 

      Tensor forward(const Tensor& input) override {
        Tensor output = input; // Copie les dimensions et données
        
        // Appliquer sigmoid élément par élément
        for (int i = 0; i < output.size(); ++i) {
            float x = output[i];
            output[i] = 1.0f / (1.0f + std::exp(-x));
        }
        
        return output;
    }
    
    Tensor backward(const Tensor& gradOutput) override {
        // Placeholder implementation
        Tensor gradInput = gradOutput; // Copie les dimensions et données
        
        // Calculer le gradient en utilisant la sortie du forward pass
        for (int i = 0; i < gradInput.size(); ++i) {
            float sigmoid_x = 1.0f / (1.0f + std::exp(-gradInput[i]));
            gradInput[i] = gradOutput[i] * sigmoid_x * (1 - sigmoid_x);
        }
        
        return gradInput;
    }    

    std::string getName() const override {
        return "Sigmoid";
    }
};


class SoftmaxLayer : public Layer {
private:
    Tensor output_cache; // Cache pour la rétropropagation
    
public:
    SoftmaxLayer() = default;
    ~SoftmaxLayer() override = default;
    
    Tensor forward(const Tensor& input) override {
        int batch_size = input.dim(0);
        int channels = input.dim(1);
        int height = input.dim(2);
        int width = input.dim(3);
        
        // Softmax est généralement appliqué sur la dimension des canaux
        // On suppose que l'input est de shape [batch, classes, 1, 1] pour classification
        if (height != 1 || width != 1) {
            throw std::runtime_error("SoftmaxLayer: L'input devrait avoir height=1 et width=1 pour la classification");
        }
        
        Tensor output(batch_size, channels, height, width);
        output_cache = Tensor(batch_size, channels, height, width);
        
        #pragma omp parallel for
        for (int b = 0; b < batch_size; ++b) {
            // Trouver la valeur maximale pour la stabilité numérique
            float max_val = -std::numeric_limits<float>::infinity();
            for (int c = 0; c < channels; ++c) {
                max_val = std::max(max_val, input(b, c, 0, 0));
            }
            
            // Calculer la somme des exponentielles
            float sum_exp = 0.0f;
            for (int c = 0; c < channels; ++c) {
                float exp_val = std::exp(input(b, c, 0, 0) - max_val);
                sum_exp += exp_val;
                output_cache(b, c, 0, 0) = exp_val; // Stocker avant normalisation
            }
            
            // Normalisation
            for (int c = 0; c < channels; ++c) {
                output(b, c, 0, 0) = output_cache(b, c, 0, 0) / sum_exp;
            }
        }
        
        return output;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        int batch_size = grad_output.dim(0);
        int channels = grad_output.dim(1);
        
        Tensor grad_input(batch_size, channels, 1, 1);
        
        #pragma omp parallel for
        for (int b = 0; b < batch_size; ++b) {
            // Calculer le Jacobien pour chaque échantillon
            // Le Jacobien de softmax est: J = diag(s) - s * s^T
            // où s est le vecteur softmax
            
            // Pour efficacité, on calcule directement le produit avec le gradient
            // grad_input = J * grad_output = s * (grad_output - sum(s_i * grad_output_i))
            
            // Calculer sum(s_i * grad_output_i)
            float dot_product = 0.0f;
            for (int c = 0; c < channels; ++c) {
                dot_product += output_cache(b, c, 0, 0) * grad_output(b, c, 0, 0);
            }
            
            // Calculer le gradient
            for (int c = 0; c < channels; ++c) {
                float s = output_cache(b, c, 0, 0);
                grad_input(b, c, 0, 0) = s * (grad_output(b, c, 0, 0) - dot_product);
            }
        }
        
        return grad_input;
    }
    
    std::string getName() const override {
        return "Softmax";
    }
    
    // Méthode utilitaire pour obtenir les prédictions (classes)
    Tensor getPredictions() const {
        int batch_size = output_cache.dim(0);
        int channels = output_cache.dim(1);
        Tensor predictions(batch_size, 1, 1, 1);
        
        for (int b = 0; b < batch_size; ++b) {
            int max_class = 0;
            float max_prob = 0.0f;
            
            for (int c = 0; c < channels; ++c) {
                if (output_cache(b, c, 0, 0) > max_prob) {
                    max_prob = output_cache(b, c, 0, 0);
                    max_class = c;
                }
            }
            
            predictions(b, 0, 0, 0) = static_cast<float>(max_class);
        }
        
        return predictions;
    }
    
    // Méthode pour obtenir les probabilités
    const Tensor& getProbabilities() const {
        return output_cache;
    }
};
