#pragma once
#include "Layer.hpp"
#include "SparseTensor.hpp"
#include "SparseConvLayer3D.hpp"
#include "LayerNorm3DSparse.hpp"

// ---------------------------------------------------------------------------
// PatchifyStem3DSparse
//
// Convertit le tenseur d'entrée brutalement en SparseTensor (threshold=0 pour 
// conserver tous les pixels > 0 de l'IRM/CT), puis applique une SparseConv
// 4x4x4 avec stride=4. Enfin, normalise en gardant les features sparse avant 
// de rendre un Tensor Dense pour CNN.
// ---------------------------------------------------------------------------
class PatchifyStem3DSparse : public Layer {
public:
    PatchifyStem3DSparse(int in_channels, int out_channels, int patch_size = 4, float threshold = 0.0f)
        : threshold_(threshold)
    {
        // submanifold = false (car the stride is > 1)
        conv_ = std::make_shared<SparseConvLayer3D>(
            in_channels, out_channels,
            patch_size, patch_size, patch_size,
            patch_size, patch_size, patch_size,
            0, 0, 0, false);
        ln_ = std::make_shared<LayerNorm3DSparse>(out_channels);
        isTrainable = true;
    }

    std::string getName() const override { return "PatchifyStem3DSparse"; }

    int numParams() const override {
        return conv_->getWeights().size() + conv_->getBias().size() + ln_->numParams();
    }

    Tensor forward(const Tensor& input) override {
        input_shape_cache_ = input.shape();
        
        SparseTensor sp = SparseTensor::from_dense(input, threshold_);
        sp_cache_out_conv_ = conv_->forward(sp); // SparseConv returns SparseTensor

        SparseTensor sp_ln = ln_->forward(sp_cache_out_conv_);
        return sp_ln.to_dense();
    }

    Tensor backward(const Tensor& gradOutput) override {
        // 1. Grad dense -> Grad Sparse (seules positions actives de la sortie LNorm)
        // Les coordonnées du SparseTensor de sortie sont celles de sp_cache_out_conv_
        const int N = sp_cache_out_conv_.nnz();
        const int C = sp_cache_out_conv_.num_channels;

        SparseTensor sp_grad_ln;
        sp_grad_ln.batch_size = sp_cache_out_conv_.batch_size;
        sp_grad_ln.num_channels = C;
        sp_grad_ln.spatial_d = sp_cache_out_conv_.spatial_d;
        sp_grad_ln.spatial_h = sp_cache_out_conv_.spatial_h;
        sp_grad_ln.spatial_w = sp_cache_out_conv_.spatial_w;
        sp_grad_ln.coords = sp_cache_out_conv_.coords;
        sp_grad_ln.features.resize(N, C);

        for (int i = 0; i < N; ++i) {
            const int b = sp_grad_ln.coords(i, 0);
            const int d = sp_grad_ln.coords(i, 1);
            const int h = sp_grad_ln.coords(i, 2);
            const int w = sp_grad_ln.coords(i, 3);
            for (int c = 0; c < C; ++c) {
                sp_grad_ln.features(i, c) = gradOutput(b, c, d, h, w);
            }
        }

        // 2. Backward LayerNorm Sparse
        SparseTensor sp_grad_conv = ln_->backward(sp_grad_ln);

        // 3. Backward SparseConv
        SparseTensor sp_grad_in = conv_->backward(sp_grad_conv);

        // 4. Sparse -> Dense pour l'entrée
        return sp_grad_in.to_dense();
    }

    void updateParams(Optimizer& optimizer) override {
        conv_->updateParams(optimizer);
        ln_->updateParams(optimizer);
    }

    void saveParameters(boost::archive::binary_oarchive& archive) const override {
        archive << conv_->getWeights() << conv_->getBias();
        ln_->saveParameters(archive);
    }

    void loadParameters(boost::archive::binary_iarchive& archive) override {
        archive >> conv_->getWeights() >> conv_->getBias();
        ln_->loadParameters(archive);
    }

private:
    float threshold_;
    std::vector<int> input_shape_cache_;
    SparseTensor sp_cache_out_conv_;

    std::shared_ptr<SparseConvLayer3D> conv_;
    std::shared_ptr<LayerNorm3DSparse> ln_;
};
