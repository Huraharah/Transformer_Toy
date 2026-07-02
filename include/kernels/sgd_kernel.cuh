#pragma once

#include <cstddef>

void launchSGDUpdateKernel(
    float* values,
    const float* grads,
    size_t size,
    float learningRate,
    float weightDecay
);