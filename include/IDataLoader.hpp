#pragma once
#include <utility>
#include "Tensor.hpp"

// =============================================================================
// Interface commune à DataLoader et DataLoader3D
// =============================================================================
class IDataLoader {
public:
    virtual ~IDataLoader() = default;

    virtual void reset()                                                      = 0;
    virtual bool hasNext()                                              const = 0;
    virtual std::pair<Tensor, Tensor> nextBatch()                             = 0;
    virtual float computeAccuracy(const Tensor& predictions,
                                  const Tensor& targets)               const  = 0;
    virtual int getNumBatches() const                                         = 0;
    virtual int getBatchSize()  const                                         = 0;
    virtual int getNumSamples() const                                         = 0;
    virtual void setMaxSamples(int max_samples)                               = 0;
};