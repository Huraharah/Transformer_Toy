#pragma once

#include <cstddef>


void launchEmbeddingForward(
	const float* tokenIds,
	const float* table,
	float* output,
	size_t batchSize,
	size_t sequenceLength,
	size_t embeddingDim,
	size_t vocabSize
);

void launchEmbeddingBackward(
    const float* tokenIds,
    const float* gradOutput,
    float* tableGrad,
    size_t batchSize,
    size_t sequenceLength,
    size_t embeddingDim,
    size_t vocabSize
);