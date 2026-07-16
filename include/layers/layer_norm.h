#pragma once

#include <vector>

#include "core/tensor.h"
#include "core/parameter.h"
#include "training/training_profiler.h"

class TrainingProfiler;

class LayerNorm {
private:
    size_t featureDim_;
    float epsilon_;

    Parameter gamma_;   // scale
    Parameter beta_;    // shift

    Tensor cachedInput_;
    Tensor cachedMean_;
    Tensor cachedInvStd_;

    TrainingProfiler* profiler_ = nullptr;

public:
    LayerNorm(size_t featureDim, float epsilon = 1e-5f);

    Tensor forward(const Tensor& input);
	Tensor backward(const Tensor& gradOutput);

    void setProfiler(TrainingProfiler* profiler);

    std::vector<Parameter*> parameters();
};