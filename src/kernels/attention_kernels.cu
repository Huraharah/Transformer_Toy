#include "kernels/attention_kernels.cuh"
#include "core/cuda_utils.h"

#include <cmath>

namespace kernels {
    __global__ void selfAttentionWeightsForwardKernel(
        const float* q,
        const float* k,
        float* attentionWeights,
        size_t batchSize,
        size_t sequenceLength,
        size_t embedDim
    ) {
        size_t b = blockIdx.x;
        size_t t = blockIdx.y;

        if (b >= batchSize || t >= sequenceLength) {
            return;
        }

        float scale = rsqrtf(static_cast<float>(embedDim));
        float maxScore = -3.402823466e+38F;

        for (size_t j = 0; j <= t; ++j) {
            float score = 0.0f;

            for (size_t f = 0; f < embedDim; ++f) {
                score +=
                    q[(b * sequenceLength + t) * embedDim + f] *
                    k[(b * sequenceLength + j) * embedDim + f];
            }

            score *= scale;

            if (score > maxScore) {
                maxScore = score;
            }
        }

        float sumExp = 0.0f;

        for (size_t j = 0; j <= t; ++j) {
            float score = 0.0f;

            for (size_t f = 0; f < embedDim; ++f) {
                score +=
                    q[(b * sequenceLength + t) * embedDim + f] *
                    k[(b * sequenceLength + j) * embedDim + f];
            }

            score *= scale;

            float weight = expf(score - maxScore);

            size_t weightIndex =
                (b * sequenceLength + t) * sequenceLength + j;

            attentionWeights[weightIndex] = weight;
            sumExp += weight;
        }

        for (size_t j = 0; j < sequenceLength; ++j) {
            size_t weightIndex =
                (b * sequenceLength + t) * sequenceLength + j;

            if (j <= t) {
                attentionWeights[weightIndex] /= sumExp;
            }
            else {
                attentionWeights[weightIndex] = 0.0f;
            }
        }
    }

    __global__ void selfAttentionValuesForwardKernel(
        const float* v,
        const float* droppedAttentionWeights,
        float* output,
        size_t batchSize,
        size_t sequenceLength,
        size_t embedDim
    ) {
        size_t b = blockIdx.x;
        size_t t = blockIdx.y;

        if (b >= batchSize || t >= sequenceLength) {
            return;
        }

        size_t attentionRowBase =
            (b * sequenceLength + t) * sequenceLength;

        for (size_t f = 0; f < embedDim; ++f) {
            float sum = 0.0f;

            for (size_t j = 0; j <= t; ++j) {
                sum +=
                    droppedAttentionWeights[attentionRowBase + j] *
                    v[(b * sequenceLength + j) * embedDim + f];
            }

            output[(b * sequenceLength + t) * embedDim + f] = sum;
        }
    }

    __global__ void selfAttentionBackwardValuesKernel(
        const float* v,
        const float* droppedAttentionWeights,
        const float* gradOutput,
        float* gradDroppedWeights,
        float* gradV,
        size_t batchSize,
        size_t sequenceLength,
        size_t embedDim
    ) {
        size_t b = blockIdx.x;
        size_t t = blockIdx.y;
        size_t threadId = threadIdx.x;

        if (b >= batchSize || t >= sequenceLength) {
            return;
        }

        size_t queryBase =
            (b * sequenceLength + t) * embedDim;

        size_t attentionRowBase =
            (b * sequenceLength + t) * sequenceLength;

        for (size_t j = threadId; j <= t; j += blockDim.x) {
            size_t keyValueBase =
                (b * sequenceLength + j) * embedDim;

            float gradWeight = 0.0f;

            for (size_t f = 0; f < embedDim; ++f) {
                float gradOut =
                    gradOutput[queryBase + f];

                gradWeight +=
                    gradOut *
                    v[keyValueBase + f];

                atomicAdd(
                    &gradV[keyValueBase + f],
                    droppedAttentionWeights[attentionRowBase + j] *
                    gradOut
                );
            }

            gradDroppedWeights[attentionRowBase + j] =
                gradWeight;
        }
    }

    __global__ void selfAttentionBackwardWeightsKernel(
        const float* q,
        const float* k,
        const float* attentionWeights,
        const float* gradAttentionWeights,
        float* gradQ,
        float* gradK,
        size_t batchSize,
        size_t sequenceLength,
        size_t embedDim
    ) {
        size_t b = blockIdx.x;
        size_t t = blockIdx.y;
        size_t threadId = threadIdx.x;

        if (b >= batchSize || t >= sequenceLength) {
            return;
        }

        size_t queryBase =
            (b * sequenceLength + t) * embedDim;

        size_t attentionRowBase =
            (b * sequenceLength + t) * sequenceLength;

        float scale =
            rsqrtf(static_cast<float>(embedDim));

        extern __shared__ float reduction[];

        float localWeightedSum = 0.0f;

        for (size_t j = threadId; j <= t; j += blockDim.x) {
            localWeightedSum +=
                gradAttentionWeights[attentionRowBase + j] *
                attentionWeights[attentionRowBase + j];
        }

        reduction[threadId] = localWeightedSum;
        __syncthreads();

        for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadId < stride) {
                reduction[threadId] +=
                    reduction[threadId + stride];
            }

            __syncthreads();
        }

        float weightedSum = reduction[0];

        for (size_t j = threadId; j <= t; j += blockDim.x) {
            size_t keyBase =
                (b * sequenceLength + j) * embedDim;

            float gradScore =
                attentionWeights[attentionRowBase + j] *
                (
                    gradAttentionWeights[attentionRowBase + j] -
                    weightedSum
                    ) *
                scale;

            for (size_t f = 0; f < embedDim; ++f) {
                atomicAdd(
                    &gradQ[queryBase + f],
                    gradScore * k[keyBase + f]
                );

                atomicAdd(
                    &gradK[keyBase + f],
                    gradScore * q[queryBase + f]
                );
            }
        }
    }

    __global__ void multiHeadAttentionWeightsForwardKernel(
        const float* q,
        const float* k,
        float* attentionWeights,
        size_t batchSize,
        size_t sequenceLength,
        size_t numHeads,
        size_t headDim
    ) {
        size_t b = blockIdx.x;
        size_t h = blockIdx.y;
        size_t t = blockIdx.z;

        if (b >= batchSize || h >= numHeads || t >= sequenceLength) {
            return;
        }

        size_t embedDim = numHeads * headDim;
        float scale = rsqrtf(static_cast<float>(headDim));

        float maxScore = -3.402823466e+38F;

        for (size_t j = 0; j <= t; ++j) {
            float score = 0.0f;

            for (size_t f = 0; f < headDim; ++f) {
                size_t globalF = h * headDim + f;

                score +=
                    q[(b * sequenceLength + t) * embedDim + globalF] *
                    k[(b * sequenceLength + j) * embedDim + globalF];
            }

            score *= scale;

            if (score > maxScore) {
                maxScore = score;
            }
        }

        float sumExp = 0.0f;

        for (size_t j = 0; j <= t; ++j) {
            float score = 0.0f;

            for (size_t f = 0; f < headDim; ++f) {
                size_t globalF = h * headDim + f;

                score +=
                    q[(b * sequenceLength + t) * embedDim + globalF] *
                    k[(b * sequenceLength + j) * embedDim + globalF];
            }

            score *= scale;

            float weight = expf(score - maxScore);

            attentionWeights[
                ((b * numHeads + h) * sequenceLength + t) * sequenceLength + j
            ] = weight;

            sumExp += weight;
        }

        for (size_t j = 0; j < sequenceLength; ++j) {
            size_t weightIndex =
                ((b * numHeads + h) * sequenceLength + t) * sequenceLength + j;

            if (j <= t) {
                attentionWeights[weightIndex] /= sumExp;
            }
            else {
                attentionWeights[weightIndex] = 0.0f;
            }
        }
    }

    __global__ void multiHeadAttentionValuesForwardKernel(
        const float* v,
        float* attentionWeights,
        float* output,
        size_t batchSize,
        size_t sequenceLength,
        size_t numHeads,
        size_t headDim
        ) {

        size_t b = blockIdx.x;
        size_t h = blockIdx.y;
        size_t t = blockIdx.z;

        if (b >= batchSize || h >= numHeads || t >= sequenceLength) {
            return;
        }

        size_t embedDim = offsetof * headDim;

        for (size_t f = 0; f < headDim; ++f) {
            size_t globalF = h * headDim + f;

            float sum = 0.0f;

            for (size_t j = 0; j <= t; ++j) {
                float weight =
                    attentionWeights[
                        ((b * numHeads + h) * sequenceLength + t) * sequenceLength + j
                    ];

                sum +=
                    weight *
                    v[(b * sequenceLength + j) * embedDim + globalF];
            }

            output[(b * sequenceLength + t) * embedDim + globalF] = sum;
        }
    }

    __global__ void multiHeadAttentionBackwardValuesKernel(
        const float* v,
        const float* droppedAttentionWeights,
        const float* gradOutput,
        float* gradDroppedWeights,
        float* gradV,
        size_t batchSize,
        size_t sequenceLength,
        size_t numHeads,
        size_t headDim
    ) {
        size_t b = blockIdx.x;
        size_t h = blockIdx.y;
        size_t t = blockIdx.z;
        size_t threadId = threadIdx.x;

        if (b >= batchSize || h >= numHeads || t >= sequenceLength) {
            return;
        }

        size_t embedDim = numHeads * headDim;
        size_t queryBase = (b * sequenceLength + t) * embedDim;
        size_t attentionRowBase = ((b * numHeads + h) * sequenceLength + t) * sequenceLength;

        for (size_t j = threadId; j <= t; j += blockDim.x) {
            size_t keyValueBase = (b * sequenceLength + j) * embedDim;
            float gradWeight = 0.0f;

            for (size_t f = 0; f < headDim; ++f) {
                size_t globalF = h * headDim + f;
                float gradOut = gradOutput[queryBase + globalF];

                gradWeight += gradOut * v[keyValueBase + globalF];

                atomicAdd(
                    &gradV[keyValueBase + globalF],
                    droppedAttentionWeights[attentionRowBase + j] * gradOut
                );
            }

            gradDroppedWeights[attentionRowBase + j] = gradWeight;
        }
    }

    __global__ void multiHeadAttentionBackwardWeightsKernel(
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
    ) {
        size_t b = blockIdx.x;
        size_t h = blockIdx.y;
        size_t t = blockIdx.z;
        size_t threadId = threadIdx.x;

        if (b >= batchSize || h >= numHeads || t >= sequenceLength) {
            return;
        }

        size_t embedDim = numHeads * headDim;
        size_t queryBase = (b * sequenceLength + t) * embedDim;
        size_t attentionRowBase = ((b * numHeads + h) * sequenceLength + t) * sequenceLength;

        float scale = rsqrtf(static_cast<float>(headDim));

        extern __shared__ float reduction[];

        float localWeightedSum = 0.0f;

        for (size_t j = threadId; j <= t; j += blockDim.x) {
            localWeightedSum +=
                gradAttentionWeights[attentionRowBase + j] *
                attentionWeights[attentionRowBase + j];
        }

        reduction[threadId] = localWeightedSum;
        __syncthreads();

        for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadId < stride) {
                reduction[threadId] += reduction[threadId + stride];
            }

            __syncthreads();
        }

        float weightedSum = reduction[0];

        for (size_t j = threadId; j <= t; j += blockDim.x) {
            size_t keyValueBase = (b * sequenceLength + j) * embedDim;

            float gradScore =
                attentionWeights[attentionRowBase + j] *
                (
                    gradAttentionWeights[attentionRowBase + j] -
                    weightedSum
                    ) *
                scale;

            for (size_t f = 0; f < headDim; ++f) {
                size_t globalF = h * headDim + f;

                atomicAdd(
                    &gradQ[queryBase + globalF],
                    gradScore * k[keyValueBase + globalF]
                );

                atomicAdd(
                    &gradK[keyValueBase + globalF],
                    gradScore * q[queryBase + globalF]
                );
            }
        }
    }
}

void launchSelfAttentionWeightsForward(
    const float* q,
    const float* k,
    float* attentionWeights,
    size_t batchSize,
    size_t sequenceLength,
    size_t embedDim
) {
    if (batchSize == 0 || sequenceLength == 0 || embedDim == 0) {
        return;
    }

    dim3 grid(
        static_cast<unsigned int>(batchSize),
        static_cast<unsigned int>(sequenceLength)
    );

    dim3 block(1);

    kernels::selfAttentionWeightsForwardKernel << <grid, block >> > (
        q,
        k,
        attentionWeights,
        batchSize,
        sequenceLength,
        embedDim
        );

    CUDA_CHECK(cudaGetLastError());
}

void launchSelfAttentionValuesForward(
    const float* v,
    const float* droppedAttentionWeights,
    float* output,
    size_t batchSize,
    size_t sequenceLength,
    size_t embedDim
) {
    if (batchSize == 0 || sequenceLength == 0 || embedDim == 0) {
        return;
    }

    dim3 grid(
        static_cast<unsigned int>(batchSize),
        static_cast<unsigned int>(sequenceLength)
    );

    dim3 block(1);

    kernels::selfAttentionValuesForwardKernel << <grid, block >> > (
        v,
        droppedAttentionWeights,
        output,
        batchSize,
        sequenceLength,
        embedDim
        );

    CUDA_CHECK(cudaGetLastError());
}

void launchSelfAttentionBackwardValues(
    const float* v,
    const float* droppedAttentionWeights,
    const float* gradOutput,
    float* gradDroppedWeights,
    float* gradV,
    size_t batchSize,
    size_t sequenceLength,
    size_t embedDim
) {
    constexpr unsigned int maximumThreads = 256;

    unsigned int threads = 1;

    while (threads < sequenceLength && threads < maximumThreads) {
        threads <<= 1;
    }

    dim3 grid(
        static_cast<unsigned int>(batchSize),
        static_cast<unsigned int>(sequenceLength)
    );

    dim3 block(threads);

    kernels::selfAttentionBackwardValuesKernel << <grid, block >> > (
        v,
        droppedAttentionWeights,
        gradOutput,
        gradDroppedWeights,
        gradV,
        batchSize,
        sequenceLength,
        embedDim
        );

    CUDA_CHECK(cudaGetLastError());
}

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
) {
    constexpr unsigned int maximumThreads = 256;

    unsigned int threads = 1;

    while (threads < sequenceLength && threads < maximumThreads) {
        threads <<= 1;
    }

    dim3 grid(
        static_cast<unsigned int>(batchSize),
        static_cast<unsigned int>(sequenceLength)
    );

    dim3 block(threads);

    size_t sharedMemoryBytes =
        threads * sizeof(float);

    kernels::selfAttentionBackwardWeightsKernel << <
        grid,
        block,
        sharedMemoryBytes
        >> > (
            q,
            k,
            attentionWeights,
            gradAttentionWeights,
            gradQ,
            gradK,
            batchSize,
            sequenceLength,
            embedDim
            );

    CUDA_CHECK(cudaGetLastError());
}

void launchMultiHeadAttentionWeightsForward(
    const float* q,
    const float* k,
    float* attentionWeights,
    size_t batchSize,
    size_t sequenceLength,
    size_t numHeads,
    size_t headDim
) {
    if (
        batchSize == 0 ||
        sequenceLength == 0 ||
        numHeads == 0 ||
        headDim == 0
        ) {
        return;
    }

    dim3 grid(
        static_cast<unsigned int>(batchSize),
        static_cast<unsigned int>(numHeads),
        static_cast<unsigned int>(sequenceLength)
    );

    dim3 block(1);

    kernels::multiHeadAttentionWeightsForwardKernel<<<grid, block>>>(
        q,
        k,
        attentionWeights,
        batchSize,
        sequenceLength,
        numHeads,
        headDim
        );

    CUDA_CHECK(cudaGetLastError());    
}

void launchMultiHeadAttentionValuesForward(
    const float* v,
    float* attentionWeights,
    float* output,
    size_t batchSize,
    size_t sequenceLength,
    size_t numHeads,
    size_t headDim
) {
    if (
        batchSize == 0 ||
        sequenceLength == 0 ||
        numHeads == 0 ||
        headDim == 0
        ) {
        return;
    }

    dim3 grid(
        static_cast<unsigned int>(batchSize),
        static_cast<unsigned int>(numHeads),
        static_cast<unsigned int>(sequenceLength)
    );

    dim3 block(1);

    kernels::multiHeadAttentionValuesForwardKernel<<<grid, block>>>(
        v,
        attentionWeights,
        output,
        batchSize,
        sequenceLength,
        numHeads,
        headDim
        );

    CUDA_CHECK(cudatGetLastError());
}

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
) {
    constexpr unsigned int maximumThreads = 256;

    unsigned int threads = 1;

    while (threads < sequenceLength && threads < maximumThreads) {
        threads <<= 1;
    }

    if (threads > maximumThreads) {
        threads = maximumThreads;
    }

    dim3 grid(
        static_cast<unsigned int>(batchSize),
        static_cast<unsigned int>(numHeads),
        static_cast<unsigned int>(sequenceLength)
    );

    dim3 block(threads);

    kernels::multiHeadAttentionBackwardValuesKernel<<<grid, block>>>(
            v,
            droppedAttentionWeights,
            gradOutput,
            gradDroppedWeights,
            gradV,
            batchSize,
            sequenceLength,
            numHeads,
            headDim
            );

    CUDA_CHECK(cudaGetLastError());
}

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
) {
    constexpr unsigned int maximumThreads = 256;

    unsigned int threads = 1;

    while (threads < sequenceLength && threads < maximumThreads) {
        threads <<= 1;
    }

    if (threads > maximumThreads) {
        threads = maximumThreads;
    }

    dim3 grid(
        static_cast<unsigned int>(batchSize),
        static_cast<unsigned int>(numHeads),
        static_cast<unsigned int>(sequenceLength)
    );

    dim3 block(threads);

    size_t sharedMemoryBytes =
        threads * sizeof(float);

    kernels::multiHeadAttentionBackwardWeightsKernel<<<grid, block, sharedMemoryBytes>>>(
        q,
        k,
        attentionWeights,
        gradAttentionWeights,
        gradQ,
        gradK,
        batchSize,
        sequenceLength,
        numHeads,
        headDim
        );

    CUDA_CHECK(cudaGetLastError());
}
