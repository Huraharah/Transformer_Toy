#pragma once
#include "core/tensor.h"

namespace LayerUtils {

    Tensor flatten3DTo2D(const Tensor& input);

    Tensor unflatten2DTo3D(
        const Tensor& input,
        size_t batchSize,
        size_t sequenceLength
    );

}