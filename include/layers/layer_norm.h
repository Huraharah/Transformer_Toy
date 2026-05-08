#pragma once

#include "core/tensor.h"

class LayerNorm {
private:
    size_t featureDim_;
    float epsilon_;

    Tensor gamma_; // scale
    Tensor beta_;  // shift

public:
    LayerNorm(size_t featureDim, float epsilon = 1e-5f);

    Tensor forward(const Tensor& input) const;

    const Tensor& gamma() const;
    const Tensor& beta() const;
};