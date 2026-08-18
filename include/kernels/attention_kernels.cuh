#pragma once
#include "core/cuda_utils.h"

#include <cstddef>

void launchSelfAttentionWeightsForward(
    const float* q,
    const float* k,
    float* attentionWeights,
    size_t batchSize,
    size_t sequenceLength,
    size_t embedDim
);

void launchSelfAttentionValuesForward(
    const float* v,
    const float* droppedAttentionWeights,
    float* output,
    size_t batchSize,
    size_t sequenceLength,
    size_t embedDim
);

void launchSelfAttentionBackwardValues(
    const float* v,
    const float* droppedAttentionWeights,
    const float* gradOutput,
    float* gradDroppedWeights,
    float* gradV,
    size_t batchSize,
    size_t sequenceLength,
    size_t embedDim
);

void launchSelfAttentionBackwardWeights(
    const float* q,
    const float* k,
    const float* attentionWeights,
    const float* gradAttentionWeights,
    float* gradQ,
    float* gradK,
    size_t batchSize,
    size_t sequenceLength,
    size_t embedDim
);

void launchMultiHeadAttentionWeightsForward(
    const float* q,
    const float* k,
    float* attentionWeights,
    size_t batchSize,
    size_t sequenceLength,
    size_t numHeads,
    size_t headDim
);

void launchMultiHeadAttentionValuesForward(
    const float* v,
    float* attentionWeights,
    float* output,
    size_t batchSize,
    size_t sequenceLength,
    size_t numHeads,
    size_t headDim
);

void launchMultiHeadAttentionBackwardValues(
    const float* v,
    const float* droppedAttentionWeights,
    const float* gradOutput,
    float* gradDroppedWeights,
    float* gradV,
    size_t batchSize,
    size_t sequenceLength,
    size_t numHeads,
    size_t headDim
);

void launchMultiHeadAttentionBackwardWeights(
    const float* q,
    const float* k,
    const float* attentionWeights,
    const float* gradAttentionWeights,
    float* gradQ,
    float* gradK,
    size_t batchSize,
    size_t sequenceLength,
    size_t numHeads,
    size_t headDim
);