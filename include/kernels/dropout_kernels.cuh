// dropout_kernels.cuh

#pragma once

#include "core/cuda_utils.h"

void launchDropoutForward(
    const float* input,
    const float* mask,
    float* output,
    size_t size
);

void launchDropoutBackward(
    const float* gradOutput,
    const float* mask,
    float* gradInput,
    size_t size
);