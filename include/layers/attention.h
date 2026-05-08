#pragma once

#include "core/tensor.h"
#include "core/random.h"
#include "layers/linear.h"

class SelfAttention {
private:
    size_t embedDim_;

    Linear queryProj_;
    Linear keyProj_;
    Linear valueProj_;
    Linear outputProj_;

    Tensor flatten3DTo2D(const Tensor& input) const;
    Tensor unflatten2DTo3D(const Tensor& input, size_t batchSize, size_t sequenceLength) const;

public:
    SelfAttention(size_t embedDim, Random& rng);

    Tensor forward(const Tensor& input) const;
};