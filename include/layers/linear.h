#pragma once

#include "core/tensor.h"
#include "core/random.h"

class Linear {
private:
    size_t inFeatures_;
    size_t outFeatures_;

    Tensor weights_; // [outFeatures, inFeatures]
    Tensor bias_;    // [outFeatures]

public:
    Linear(size_t inFeatures, size_t outFeatures, Random& rng);

    Tensor forward(const Tensor& input) const;

    const Tensor& weights() const;
    const Tensor& bias() const;
};