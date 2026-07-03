#pragma once
#include "core/cuda_utils.h"
#include "kernels/kernel_config.cuh"

#include <cstddef>

void launchLinearForward(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    size_t batchSize,
    size_t inFeatures,
    size_t outFeatures,
	bool useBias = true
);