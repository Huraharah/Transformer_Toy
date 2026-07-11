#pragma once

#include "optimizer.h"

class SGDOptimizer : public Optimizer {
private:
    float learningRate;
    float weightDecay;

public:
    explicit SGDOptimizer(float learningRate_, float weightDecay_ = 0.0f);

    void step(std::vector<Parameter*>& parameters) override;

    float getLearningRate() const override;
    void setLearningRate(float learningRate) override;
};