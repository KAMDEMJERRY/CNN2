#pragma once
#include "Layer.hpp"
#include "SparseTensor.hpp"
#include "DepthwiseSparseConv3D.hpp"
#include "LayerNorm3DSparse.hpp"
#include "ActivationLayer.hpp" // for gelu_approx / gelu_grad / gelu_inplace
#include <random>

// ---------------------------------------------------------------------------
// ConvNeXtBlock3DSparse
//
// Bloc ConvNeXt intégralement sparse.
// Entrée Dense -> Sparse -> DWConv -> LN -> Expand -> GELU -> Contract -> Dense.
// Poids: DWConv, LN, Expand(C->4C), Contract(4C->C).
// ---------------------------------------------------------------------------
class ConvNeXtBlock3DSparse : public Layer {
public:
    ConvNeXtBlock3DSparse(int channels, int kernel_size = 7, float layer_scale_init = 1e-6f, float threshold = 0.0f)
        : channels_(channels), threshold_(threshold)
    {
        dwconv_ = std::make_shared<DepthwiseSparseConv3D>(channels, kernel_size);
        ln_ = std::make_shared<LayerNorm3DSparse>(channels);

        int ex_channels = 4 * channels;

        W_ex_ = Eigen::MatrixXf::Zero(channels, ex_channels);
        b_ex_ = Eigen::VectorXf::Zero(ex_channels);
        dW_ex_= Eigen::MatrixXf::Zero(channels, ex_channels);
        db_ex_= Eigen::VectorXf::Zero(ex_channels);

        W_co_ = Eigen::MatrixXf::Zero(ex_channels, channels);
        b_co_ = Eigen::VectorXf::Zero(channels);
        dW_co_= Eigen::MatrixXf::Zero(ex_channels, channels);
        db_co_= Eigen::VectorXf::Zero(channels);

        gamma_ = Eigen::VectorXf::Constant(channels, layer_scale_init);
        d_gamma_ = Eigen::VectorXf::Zero(channels);

        std::mt19937 gen{std::random_device{}()};
        float scale_ex = std::sqrt(2.0f / channels);
        std::normal_distribution<float> d_ex(0.0f, scale_ex);
        for(int i = 0; i < W_ex_.size(); ++i) *(W_ex_.data() + i) = d_ex(gen);

        float scale_co = std::sqrt(2.0f / ex_channels);
        std::normal_distribution<float> d_co(0.0f, scale_co);
        for(int i = 0; i < W_co_.size(); ++i) *(W_co_.data() + i) = d_co(gen);

        isTrainable = true;
    }

    std::string getName() const override { return "ConvNeXtBlock3DSparse"; }

    int numParams() const override {
        return dwconv_->numParams() + ln_->numParams() +
               W_ex_.size() + b_ex_.size() + 
               W_co_.size() + b_co_.size() + channels_;
    }

    Tensor forward(const Tensor& input) override {
        input_dense_shape_ = input.shape();
        
        SparseTensor sp_input = SparseTensor::from_dense(input, threshold_);
        sp_cache_input_ = sp_input; // store for skip connection

        SparseTensor sp = dwconv_->forward(sp_input);
        sp = ln_->forward(sp);
        
        // Linear Expand
        features_pre_ex_ = sp.features;
        sp.features = (sp.features * W_ex_).rowwise() + b_ex_.transpose();
        
        // GELU in-place
        features_pre_gelu_ = sp.features;
        gelu_inplace(sp.features);
        
        // Linear Contract
        features_pre_co_ = sp.features;
        sp.features = (sp.features * W_co_).rowwise() + b_co_.transpose();

        features_pre_scale_ = sp.features;

        // LayerScale + Residual
        sp.features = (sp.features.array().rowwise() * gamma_.transpose().array()) + sp_cache_input_.features.array();

        return sp.to_dense();
    }

    Tensor backward(const Tensor& gradOutput) override {
        const int N = sp_cache_input_.nnz();
        const int B = sp_cache_input_.batch_size;

        SparseTensor sp_grad_out;
        sp_grad_out.batch_size = B;
        sp_grad_out.num_channels = channels_;
        sp_grad_out.spatial_d = sp_cache_input_.spatial_d;
        sp_grad_out.spatial_h = sp_cache_input_.spatial_h;
        sp_grad_out.spatial_w = sp_cache_input_.spatial_w;
        sp_grad_out.coords = sp_cache_input_.coords;
        sp_grad_out.features.resize(N, channels_);

        // Extract active gradients
        for (int i = 0; i < N; ++i) {
            const int b = sp_grad_out.coords(i, 0);
            const int d = sp_grad_out.coords(i, 1);
            const int h = sp_grad_out.coords(i, 2);
            const int w = sp_grad_out.coords(i, 3);
            for (int c = 0; c < channels_; ++c) {
                sp_grad_out.features(i, c) = gradOutput(b, c, d, h, w);
            }
        }

        d_gamma_.setZero();
        SparseTensor gs = sp_grad_out; // Will become grad w.r.t the block main path

        if (N > 0) {
            // Backward LayerScale
            d_gamma_ = (sp_grad_out.features.array() * features_pre_scale_.array()).colwise().sum();
            gs.features = sp_grad_out.features.array().rowwise() * gamma_.transpose().array();
            
            // Backward Contract
            dW_co_ = features_pre_co_.transpose() * gs.features;
            db_co_ = gs.features.colwise().sum();
            gs.features = gs.features * W_co_.transpose();

            // Backward GELU
            gs.features = gs.features.array() * features_pre_gelu_.array().unaryExpr([](float x){ return gelu_grad(x); });

            // Backward Expand
            dW_ex_ = features_pre_ex_.transpose() * gs.features;
            db_ex_ = gs.features.colwise().sum();
            gs.features = gs.features * W_ex_.transpose();
            
            // Normalise weights grad
            dW_co_ /= static_cast<float>(B);
            db_co_ /= static_cast<float>(B);
            dW_ex_ /= static_cast<float>(B);
            db_ex_ /= static_cast<float>(B);
        }

        // Backward remaining parts (LN + DWConv)
        gs = ln_->backward(gs);
        gs = dwconv_->backward(gs);

        // Add back residual gradient
        gs.features += sp_grad_out.features;

        // to_dense backward output
        Tensor grad_input_dense(input_dense_shape_);
        grad_input_dense.setZero();
        for (int i = 0; i < N; ++i) {
            const int b = gs.coords(i, 0);
            const int d = gs.coords(i, 1);
            const int h = gs.coords(i, 2);
            const int w = gs.coords(i, 3);
            for (int c = 0; c < channels_; ++c) {
                grad_input_dense(b, c, d, h, w) = gs.features(i, c);
            }
        }

        return grad_input_dense;
    }

    void updateParams(Optimizer& optimizer) override {
        dwconv_->updateParams(optimizer);
        ln_->updateParams(optimizer);
        
        optimizer.updateWeights(W_ex_, dW_ex_);
        optimizer.updateBias(b_ex_, db_ex_);

        optimizer.updateWeights(W_co_, dW_co_);
        optimizer.updateBias(b_co_, db_co_);

        optimizer.updateBias(gamma_, d_gamma_);
    }

    void saveParameters(boost::archive::binary_oarchive& archive) const override {
        dwconv_->saveParameters(archive);
        ln_->saveParameters(archive);
        
        // Boost serialization for Eigen Matrices
        int rows_ex = W_ex_.rows(), cols_ex = W_ex_.cols();
        archive << rows_ex << cols_ex;
        for(int i=0; i<W_ex_.size(); ++i) archive << *(W_ex_.data()+i);
        archive << b_ex_;
        
        int rows_co = W_co_.rows(), cols_co = W_co_.cols();
        archive << rows_co << cols_co;
        for(int i=0; i<W_co_.size(); ++i) archive << *(W_co_.data()+i);
        archive << b_co_;
        
        archive << gamma_;
    }

    void loadParameters(boost::archive::binary_iarchive& archive) override {
        dwconv_->loadParameters(archive);
        ln_->loadParameters(archive);

        int rows, cols;
        archive >> rows >> cols;
        for(int i=0; i<W_ex_.size(); ++i) archive >> *(W_ex_.data()+i);
        archive >> b_ex_;

        archive >> rows >> cols;
        for(int i=0; i<W_co_.size(); ++i) archive >> *(W_co_.data()+i);
        archive >> b_co_;

        archive >> gamma_;
    }

private:
    int channels_;
    float threshold_;
    std::shared_ptr<DepthwiseSparseConv3D> dwconv_;
    std::shared_ptr<LayerNorm3DSparse> ln_;

    Eigen::MatrixXf W_ex_, dW_ex_;
    Eigen::VectorXf b_ex_, db_ex_;
    
    Eigen::MatrixXf W_co_, dW_co_;
    Eigen::VectorXf b_co_, db_co_;
    
    Eigen::VectorXf gamma_, d_gamma_;

    std::vector<int> input_dense_shape_;
    SparseTensor sp_cache_input_;
    Eigen::MatrixXf features_pre_ex_;
    Eigen::MatrixXf features_pre_gelu_;
    Eigen::MatrixXf features_pre_co_;
    Eigen::MatrixXf features_pre_scale_;
};
