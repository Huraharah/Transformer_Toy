#include "layers/dropout.h"
#include "core/random.h"
#include "core/tensor.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

    bool nearlyEqual(
        float a,
        float b,
        float tolerance = 1.0e-5f
    ) {
        return std::fabs(a - b) <= tolerance;
    }

    void require(
        bool condition,
        const std::string& message
    ) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

}