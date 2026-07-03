#pragma once
#include "core/cuda_utils.h"

void launchCrossEntropyForwardBackwardKernel(
    const float* logits,
    const float* targets,
    float* grad,
    float* losses,
    size_t batchSize,
    size_t numClasses
);
