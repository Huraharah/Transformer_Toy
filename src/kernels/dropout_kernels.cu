// dropout_kernels.cu

#include "kernels/dropout_kernels.cuh"
#include "core/cuda_utils.h"

namespace kernels {

    __global__ void dropoutForwardKernel(
        const float* input,
        const float* mask,
        float* output,
        size_t size
    ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= size) {
            return;
        }

        output[idx] = input[idx] * mask[idx];
    }

    __global__ void dropoutBackwardKernel(
        const float* gradOutput,
        const float* mask,
        float* gradInput,
        size_t size
    ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= size) {
            return;
        }

        gradInput[idx] = gradOutput[idx] * mask[idx];
    }

}

void launchDropoutForward(
    const float* input,
    const float* mask,
    float* output,
    size_t size
) {
    if (size == 0) {
        return;
    }

    constexpr int threads = 256;

    int blocks = static_cast<int>((size + threads - 1) / threads);

    kernels::dropoutForwardKernel<<<blocks, threads>>>(input, mask, output, size);

    CUDA_CHECK(
        cudaGetLastError()
    );
}

void launchDropoutBackward(
    const float* gradOutput,
    const float* mask,
    float* gradInput,
    size_t size
) {
    if (size == 0) {
        return;
    }

    constexpr int threads = 256;

    int blocks = static_cast<int>((size + threads - 1) / threads);

    kernels::dropoutBackwardKernel<<<blocks, threads>>>(gradOutput,mask, gradInput, size);

    CUDA_CHECK(
        cudaGetLastError()
    );
}