#pragma once

#include "core/parameter.h"

#include <vector>

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void step(std::vector<Parameter*>& parameters) = 0;

    virtual void zeroGrad(std::vector<Parameter*>& parameters) {
        for (Parameter* param : parameters) {
            if (param && param->requires_grad) {
                param->zeroGrad();
            }
        }
    }
};