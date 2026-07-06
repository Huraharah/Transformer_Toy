#pragma once
#include "core/cuda_utils.h"

#include <cstddef>

void launchSelfAttentionForward(
    const float* q,
    const float* k,
    const float* v,
    float* attentionWeights,
    float* output,
    size_t batchSize,
    size_t sequenceLength,
    size_t embedDim
);

void launchSelfAttentionBackward(
    const float* q,
    const float* k,
    const float* v,
    const float* attentionWeights,
    const float* gradOutput,
    float* gradQ,
    float* gradK,
    float* gradV,
    size_t batchSize,
    size_t sequenceLength,
    size_t embedDim
);

void launchMultiHeadAttentionForward(
    const float* q,
    const float* k,
    const float* v,
    float* attentionWeights,
    float* output,
    size_t batchSize,
    size_t sequenceLength,
    size_t numHeads,
    size_t headDim
);

void launchMultiHeadAttentionBackward(
    const float* q,
    const float* k,
    const float* v,
    const float* attentionWeights,
    const float* gradOutput,
    float* gradQ,
    float* gradK,
    float* gradV,
    size_t batchSize,
    size_t sequenceLength,
    size_t numHeads,
    size_t headDim
);