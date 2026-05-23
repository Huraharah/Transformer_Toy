#pragma once

#include "training/loss.h"

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
