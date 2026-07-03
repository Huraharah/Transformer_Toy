#pragma once

#include <vector>

#include "core/tensor.h"
#include "core/parameter.h"

class LayerNorm {
private:
    size_t featureDim_;
    float epsilon_;

    Parameter gamma_;   // scale
    Parameter beta_;    // shift

public:
    LayerNorm(size_t featureDim, float epsilon = 1e-5f);

    Tensor forward(const Tensor& input) const;

    std::vector<Parameter*> parameters();
};