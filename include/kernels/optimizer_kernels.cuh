#pragma once
#include "kernels/kernel_config.cuh"

#include <cstddef>



void launchAdamUpdateKernel(
    float* values,
    const float* grads,
    float* m,
    float* v,
    size_t size,
    float learningRate,
    float beta1,
    float beta2,
    float epsilon,
    float weightDecay,
    int timestep
);

void launchSGDUpdateKernel(
    float* values,
    const float* grads,
    size_t size,
    float learningRate,
    float weightDecay
);

