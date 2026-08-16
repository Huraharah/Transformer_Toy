#pragma once

#include "core/tensor.h"
#include "core/random.h"

class Dropout {
public:
    Dropout(float probability, Random& rng);

    Tensor forward(const Tensor& input);

    Tensor backward(const Tensor& gradOutput);

    void train();
    void eval();
    
    bool isTraining() const;
    float probability() const;

private:
    float probability_;
    float scale_;

    bool training_;

    Random& rng_;

    Tensor mask_;
    bool hasMask_;
};