#pragma once

#include "core/tensor.h"

class Loss {
public:
    virtual ~Loss() = default;

    virtual float forward(
        const Tensor& predictions,
        const Tensor& targets
    ) = 0;

    virtual Tensor backward() const = 0;
};
