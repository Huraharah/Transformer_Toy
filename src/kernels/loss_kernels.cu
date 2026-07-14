#include "kernels/loss_kernels.cuh"
#include "kernels/kernel_config.cuh"
#include "core/cuda_utils.h"

#include <stdexcept>
#include <iostream>

namespace kernels {

    __global__ void crossEntropyForwardBackwardKernel(
        const float* logits,
        const float* targets,
        float* grad,
        float* losses,
        size_t batchSize,
        size_t numClasses
    ) {
        extern __shared__ float shared[];

        size_t row = blockIdx.x;
        size_t tid = threadIdx.x;

        if (row >= batchSize) {
            return;
        }

        const float* rowLogits = logits + row * numClasses;
        float* rowGrad = grad + row * numClasses;

        int targetClass = static_cast<int>(targets[row]);

        // 1. max logit reduction
        float localMax = -3.402823466e+38F;

        for (size_t c = tid; c < numClasses; c += blockDim.x) {
            float value = rowLogits[c];
            if (value > localMax) {
                localMax = value;
            }
        }

        shared[tid] = localMax;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float other = shared[tid + stride];
                if (other > shared[tid]) {
                    shared[tid] = other;
                }
            }
            __syncthreads();
        }

        float maxLogit = shared[0];

        // 2. sum exp reduction
        float localSum = 0.0f;

        for (size_t c = tid; c < numClasses; c += blockDim.x) {
            localSum += expf(rowLogits[c] - maxLogit);
        }

        shared[tid] = localSum;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared[tid] += shared[tid + stride];
            }
            __syncthreads();
        }

        float sumExp = shared[0];
        float logSumExp = maxLogit + logf(sumExp);

        // 3. loss
        if (tid == 0) {
            if (targetClass < 0 || static_cast<size_t>(targetClass) >= numClasses) {
                losses[row] = 0.0f;
            }
            else {
                losses[row] = -rowLogits[targetClass] + logSumExp;
            }
        }

        // 4. gradient = (softmax - one_hot) / batch
        float invBatch = 1.0f / static_cast<float>(batchSize);

        for (size_t c = tid; c < numClasses; c += blockDim.x) {
            float softmax = expf(rowLogits[c] - logSumExp);
            float g = softmax;

            if (static_cast<int>(c) == targetClass) {
                g -= 1.0f;
            }

            rowGrad[c] = g * invBatch;
        }
    }

}

void launchCrossEntropyForwardBackwardKernel(
    const float* logits,
    const float* targets,
    float* grad,
    float* losses,
    size_t batchSize,
    size_t numClasses
) {
    if (batchSize == 0 || numClasses == 0) {
        return;
    }

    //if (numClasses > 4096) {
    //    std::cout << "[ERROR] Throwing invalid argument error for numClasses <= 4096" << std::endl;
    //    throw std::invalid_argument(
    //        "Current CUDA CrossEntropyLoss kernel supports numClasses <= 4096."
    //    );
    //}

    int threads = 256;

    if (numClasses < 256) {
        threads = 1;
        while (threads < static_cast<int>(numClasses)) {
            threads <<= 1;
        }
    }

    if (threads < 32) {
        threads = 32;
    }

    if (threads > 256) {
        threads = 256;
    }

    int blocks = static_cast<int>(batchSize);
    size_t sharedBytes = sizeof(float) * threads;

    //std::cout << "[DEBUG] Launching loss kernel..." << std::endl;

    kernels::crossEntropyForwardBackwardKernel<<<blocks, threads, sharedBytes>>> (
        logits,
        targets,
        grad,
        losses,
        batchSize,
        numClasses
        );

    CUDA_CHECK(cudaGetLastError());
    
}