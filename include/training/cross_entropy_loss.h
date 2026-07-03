#pragma once

#include "training/loss.h"
#include "kernels/loss_kernels.cuh"

class CrossEntropyLoss : public Loss {
private:
    Tensor cachedGrad;

public:
    float forward(
        const Tensor& logits,
        const Tensor& targets
    ) override;

    Tensor backward() const override;
};
