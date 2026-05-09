#pragma once

#include "core/tensor.h"
#include "core/random.h"
#include "layers/linear.h"
#include "core/layer_utils.h"

class SelfAttention {
private:
    size_t embedDim_;

    Linear queryProj_;
    Linear keyProj_;
    Linear valueProj_;
    Linear outputProj_;

public:
    SelfAttention(size_t embedDim, Random& rng);

    Tensor forward(const Tensor& input) const;
};