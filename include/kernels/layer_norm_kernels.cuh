#pragma once
#include "core/cuda_utils.h"

void launchLayerNormForward(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    float* means,
    float* invStds,
    size_t batchSize,
    size_t sequenceLength,
    size_t featureDim,
    float epsilon
);

void launchLayerNormBackward(
    const float* input,
    const float* gradOutput,
    const float* gamma,
    const float* means,
    const float* invStds,
    float* gradInput,
    float* gradGamma,
    float* gradBeta,
    size_t batchSize,
    size_t sequenceLength,
    size_t featureDim
);

