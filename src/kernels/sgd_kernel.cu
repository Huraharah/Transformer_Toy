#include "kernels/sgd_kernel.cuh"
#include "core/cuda_utils.h"

namespace {

    constexpr int THREADS_PER_BLOCK = 256;

    __global__ void sgdUpdateKernel(
        float* values,
        const float* grads,
        size_t size,
        float learningRate,
        float weightDecay
    ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < size) {
            float grad = grads[idx];

            if (weightDecay != 0.0f) {
                grad += weightDecay * values[idx];
            }

            values[idx] -= learningRate * grad;
        }
    }

}

void launchSGDUpdateKernel(
    float* values,
    const float* grads,
    size_t size,
    float learningRate,
    float weightDecay
) {
    if (size == 0) {
        return;
    }

    int blocks = static_cast<int>(
        (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK
        );

    sgdUpdateKernel<<<blocks, THREADS_PER_BLOCK>>>(
        values,
        grads,
        size,
        learningRate,
        weightDecay
        );

    CUDA_CHECK(cudaGetLastError());
    cudaSync();
}