#pragma once
#include "SparseTensor.hpp"
#include "Optimizer.hpp"
#include "ModelSerializer.hpp"
#include <Eigen/Dense>
#include <cmath>

// ---------------------------------------------------------------------------
//  LayerNorm3DSparse
//
//  Normalise purement la dimension channel C sur les voxels actifs uniquement.
//  Entrée/Sortie : SparseTensor
// ---------------------------------------------------------------------------
class LayerNorm3DSparse {
public:
    explicit LayerNorm3DSparse(int num_channels, float eps = 1e-6f)
        : channels_(num_channels), eps_(eps)
    {
        gamma_ = Eigen::VectorXf::Ones(num_channels);
        beta_  = Eigen::VectorXf::Zero(num_channels);
        
        d_gamma_ = Eigen::VectorXf::Zero(num_channels);
        d_beta_  = Eigen::VectorXf::Zero(num_channels);
    }

    int numParams() const { return 2 * channels_; }

    SparseTensor forward(const SparseTensor& input) {
        if (input.num_channels != channels_)
            throw std::invalid_argument("[LayerNorm3DSparse] Channel mismatch");

        const int N = input.nnz();
        SparseTensor output = input; // Copy structural info (batch, dims, coords)

        if (N == 0) return output; // Volume vide

        x_hat_cache_.resize(N, channels_);
        inv_std_cache_.resize(N);

        // Vectorisation sur les lignes [N x C]
        // mean: [N x 1]
        Eigen::VectorXf mean = input.features.rowwise().mean();

        // variance: [N x 1]
        // features - mean broadcasting implicitly non-trivial in Eigen if we don't replicate
        Eigen::MatrixXf centered = input.features.colwise() - mean;
        Eigen::VectorXf var = centered.array().square().rowwise().mean();

        inv_std_cache_ = (var.array() + eps_).rsqrt(); // [N]

        // x_hat_cache_ = centered * inv_std_cache_ (ligne par ligne)
        x_hat_cache_ = centered.array().colwise() * inv_std_cache_.array();

        // output.features = x_hat * gamma_ + beta_
        output.features = (x_hat_cache_.array().rowwise() * gamma_.transpose().array()).rowwise() + beta_.transpose().array();

        return output;
    }

    SparseTensor backward(const SparseTensor& gradOutput) {
        const int N = gradOutput.nnz();
        SparseTensor gradInput = gradOutput;
        gradInput.features.setZero();

        if (N == 0) return gradInput;

        d_gamma_.setZero();
        d_beta_.setZero();

        // Accumulation gradients
        d_gamma_ = (gradOutput.features.array() * x_hat_cache_.array()).colwise().sum();
        d_beta_  = gradOutput.features.colwise().sum();

        const float inv_C = 1.0f / channels_;

        // Gradient par rapport à l'entrée
        // dx_hat = gradOutput * gamma
        Eigen::MatrixXf dx_hat = gradOutput.features.array().rowwise() * gamma_.transpose().array();

        Eigen::VectorXf sum_dxhat = dx_hat.rowwise().sum();
        Eigen::VectorXf sum_dxhat_xhat = (dx_hat.array() * x_hat_cache_.array()).rowwise().sum();

        // gradInput = inv_std * (dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat))
        gradInput.features = dx_hat; // Start with dx_hat
        gradInput.features.colwise() -= (sum_dxhat * inv_C); // - mean(dx_hat)
        gradInput.features -= (x_hat_cache_.array().colwise() * (sum_dxhat_xhat * inv_C).array()).matrix(); // - x_hat * mean(dx_hat_xhat)
        
        gradInput.features.array().colwise() *= inv_std_cache_.array();

        return gradInput;
    }

    void updateParams(Optimizer& optimizer) {
        optimizer.updateBias(gamma_, d_gamma_);
        optimizer.updateBias(beta_, d_beta_);
        d_gamma_.setZero();
        d_beta_.setZero();
    }

    void saveParameters(boost::archive::binary_oarchive& archive) const {
        archive << gamma_ << beta_;
    }

    void loadParameters(boost::archive::binary_iarchive& archive) {
        archive >> gamma_ >> beta_;
    }

private:
    int channels_;
    float eps_;

    Eigen::VectorXf gamma_;
    Eigen::VectorXf beta_;
    Eigen::VectorXf d_gamma_;
    Eigen::VectorXf d_beta_;

    Eigen::MatrixXf x_hat_cache_;
    Eigen::VectorXf inv_std_cache_;
};
