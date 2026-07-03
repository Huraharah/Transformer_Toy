#pragma once

#include "core/tensor.h"
#include "core/random.h"
#include "core/parameter.h"

class Linear {
private:
    size_t inFeatures_;
    size_t outFeatures_;

    Parameter weights_; // [outFeatures, inFeatures]
    Parameter bias_;    // [outFeatures]

public:
    Linear(size_t inFeatures, size_t outFeatures, Random& rng);

    Tensor forward(const Tensor& input) const;

    const Tensor& weights() const;
    const Tensor& bias() const;
    std::vector<Parameter*> parameters();
};