#pragma once
#include "core/cuda_utils.h"

void launchGeluForward(
    float* data,
    size_t size
);

void launchGeluBackward(
    const float* preActivation,
    float* grad,
    size_t size
);