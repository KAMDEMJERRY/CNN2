#pragma once

#include <boost/serialization/split_free.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <Eigen/Dense>
#include "Tensor.hpp"

// Non-intrusive Boost serialization for Eigen Matrices and custom Tensor class
namespace boost {
namespace serialization {

// ─────────────────────────────────────────────────────────────────────────────
// Eigen::Matrix & Eigen::Vector Serialization
// ─────────────────────────────────────────────────────────────────────────────

template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline void save(Archive & ar, const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & m, const unsigned int version) {
    int rows = static_cast<int>(m.rows());
    int cols = static_cast<int>(m.cols());
    ar & rows;
    ar & cols;
    ar & boost::serialization::make_array(m.data(), rows * cols);
}

template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline void load(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & m, const unsigned int version) {
    int rows, cols;
    ar & rows;
    ar & cols;
    m.resize(rows, cols);
    ar & boost::serialization::make_array(m.data(), rows * cols);
}

template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline void serialize(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & m, const unsigned int version) {
    boost::serialization::split_free(ar, m, version);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tensor Serialization
// ─────────────────────────────────────────────────────────────────────────────

template<class Archive>
inline void save(Archive & ar, const Tensor & t, const unsigned int version) {
    int rank = t.ndim();
    ar & rank;
    int size = t.size();
    ar & size;
    // Save shapes
    for (int i = 0; i < rank; ++i) {
        int d = t.dim(i);
        ar & d;
    }
    // Save raw data buffer
    // make_array requires pointer to non-const for load, but save can take const pointer.
    // However depending on boost version, we might have to cast const away or it handles it.
    // To be safe, make_array handles const appropriately in output archives
    ar & boost::serialization::make_array(t.getData(), size);
}

template<class Archive>
inline void load(Archive & ar, Tensor & t, const unsigned int version) {
    int rank;
    ar & rank;
    int size;
    ar & size;
    
    std::vector<int> shape(rank);
    for (int i = 0; i < rank; ++i) {
        ar & shape[i];
    }
    
    // Reconstruct the tensor
    t = Tensor(shape);
    if (t.size() != size) {
        throw std::runtime_error("[ModelSerializer] Size mismatch during Tensor load");
    }
    
    ar & boost::serialization::make_array(t.getData(), size);
}

template<class Archive>
inline void serialize(Archive & ar, Tensor & t, const unsigned int version) {
    boost::serialization::split_free(ar, t, version);
}

} // serialization
} // boost
