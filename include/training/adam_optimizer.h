#pragma once

#include "optimizer.h"
#include "core/tensor.h"

#include <unordered_map>

class AdamOptimizer : public Optimizer {
private:
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;
    float weightDecay;

    int timestep;

    std::unordered_map<Parameter*, Tensor> m;
    std::unordered_map<Parameter*, Tensor> v;

public:
    AdamOptimizer(
        float learningRate_ = 0.001f,
        float beta1_ = 0.9f,
        float beta2_ = 0.999f,
        float epsilon_ = 1e-8f,
        float weightDecay_ = 0.0f
    );

    void step(std::vector<Parameter*>& parameters) override;
};
