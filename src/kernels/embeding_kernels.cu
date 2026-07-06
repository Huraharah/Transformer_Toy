#include "kernels/embedding_kernels.cuh"
#include "core/cuda_utils.h"

namespace kernels {

    __global__ void embeddingForwardKernel(
        const float* tokenIds,
        const float* table,
        float* output,
        size_t batchSize,
        size_t sequenceLength,
        size_t embeddingDim,
        size_t vocabSize
    ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        size_t total = batchSize * sequenceLength * embeddingDim;

        if (idx >= total) {
            return;
        }

        size_t e = idx % embeddingDim;
        size_t temp = idx / embeddingDim;
        size_t t = temp % sequenceLength;
        size_t b = temp / sequenceLength;

        int tokenId = static_cast<int>(tokenIds[b * sequenceLength + t]);

        if (tokenId < 0 || static_cast<size_t>(tokenId) >= vocabSize) {
            output[idx] = 0.0f;
            return;
        }

        output[idx] =
            table[static_cast<size_t>(tokenId) * embeddingDim + e];
    }

    __global__ void embeddingBackwardKernel(
        const float* tokenIds,
        const float* gradOutput,
        float* tableGrad,
        size_t batchSize,
        size_t sequenceLength,
        size_t embeddingDim,
        size_t vocabSize
    ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        size_t total = batchSize * sequenceLength * embeddingDim;

        if (idx >= total) {
            return;
        }

        size_t e = idx % embeddingDim;
        size_t temp = idx / embeddingDim;
        size_t t = temp % sequenceLength;
        size_t b = temp / sequenceLength;

        int tokenId = static_cast<int>(tokenIds[b * sequenceLength + t]);

        if (tokenId < 0 || static_cast<size_t>(tokenId) >= vocabSize) {
            return;
        }

        atomicAdd(
            &tableGrad[static_cast<size_t>(tokenId) * embeddingDim + e],
            gradOutput[idx]
        );
    }

}

void launchEmbeddingForward(
    const float* tokenIds,
    const float* table,
    float* output,
    size_t batchSize,
    size_t sequenceLength,
    size_t embeddingDim,
    size_t vocabSize
) {
    size_t total = batchSize * sequenceLength * embeddingDim;

    if (total == 0) {
        return;
    }

    constexpr int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    kernels::embeddingForwardKernel << <blocks, threads >> > (
        tokenIds,
        table,
        output,
        batchSize,
        sequenceLength,
        embeddingDim,
        vocabSize
        );

    CUDA_CHECK(cudaGetLastError());
    cudaSync();
}

void launchEmbeddingBackward(
    const float* tokenIds,
    const float* gradOutput,
    float* tableGrad,
    size_t batchSize,
    size_t sequenceLength,
    size_t embeddingDim,
    size_t vocabSize
) {
    size_t total = batchSize * sequenceLength * embeddingDim;

    if (total == 0) {
        return;
    }

    constexpr int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    kernels::embeddingBackwardKernel<<<blocks, threads>>>(
        tokenIds,
        gradOutput,
        tableGrad,
        batchSize,
        sequenceLength,
        embeddingDim,
        vocabSize
        );

    CUDA_CHECK(cudaGetLastError());
    cudaSync();
}