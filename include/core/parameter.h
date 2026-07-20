#pragma once

#include "core/tensor.h"

#include <string>
#include <stdexcept>

struct Parameter {
    Tensor value;
    Tensor grad;

    std::string name;
    bool requires_grad;

    Parameter()
        : requires_grad(true) {
    }

    Parameter(
        const Tensor& value_,
        const std::string& name_ = "",
        bool requires_grad_ = true
    )
        : value(value_),
        grad(value_.shape()),
        name(name_),
        requires_grad(requires_grad_) {
        zeroGrad();
    }

    void zeroGrad() {
        if (requires_grad) {
            grad.fill(0.0f);
        }
    }

    bool hasGrad() const {
        return requires_grad;
    }

    size_t size() const {
        return value.size();
    }

    void validate() const {
        if (value.size() != grad.size()) {
            throw std::runtime_error(
                "Parameter '" + name + "' has mismatched value/grad sizes."
            );
        }
    }
};