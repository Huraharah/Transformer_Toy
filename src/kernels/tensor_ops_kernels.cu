#include "kernels/tensor_ops_kernels.cuh"
#include "core/cuda_utils.h"
#include "core/tensor.h"

#include <stdexcept>

static constexpr int THREADS_PER_BLOCK = 256;

namespace kernels {
    __global__ void fillKernel(float* data, size_t size, float value) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < size) {
            data[idx] = value;
        }
    }

    __global__ void scaleKernel(float* data, size_t size, float scalar) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < size) {
            data[idx] *= scalar;
        }
    }

    __global__ void addKernel(
        const float* a,
        const float* b,
        float* out,
        size_t size
    ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < size) {
            out[idx] = a[idx] + b[idx];
        }
    }

    __global__ void flatten3DTo2DKernel(
        const float* input,
        float* output,
        size_t totalSize
    ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < totalSize) {
            output[idx] = input[idx];
        }
    }

    __global__ void unflatten2DTo3DKernel(
        const float* input,
        float* output,
        size_t totalSize
    ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < totalSize) {
            output[idx] = input[idx];
        }
    }
}

void launchTensorFill(Tensor& tensor, float value) {
    if (tensor.empty()) {
        return;
    }

    tensor.toCUDA();

    int blocks = static_cast<int>(
        (tensor.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK
        );

    kernels::fillKernel<<<blocks, THREADS_PER_BLOCK>>>(
        tensor.deviceData(),
        tensor.size(),
        value
        );

    CUDA_CHECK(cudaGetLastError());
    
}

void launchTensorScale(Tensor& tensor, float scalar) {
    if (tensor.empty()) {
        return;
    }

    tensor.toCUDA();

    int blocks = static_cast<int>(
        (tensor.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK
        );

    kernels::scaleKernel<<<blocks, THREADS_PER_BLOCK>>>(
        tensor.deviceData(),
        tensor.size(),
        scalar
        );

    CUDA_CHECK(cudaGetLastError());
    
}

void launchTensorAdd(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.size() != b.size() || a.size() != out.size()) {
        throw std::exception("tensorAddCUDA requires tensors with matching sizes.");
    }

    if (a.empty()) {
        return;
    }

    Tensor& mutableA = const_cast<Tensor&>(a);
    Tensor& mutableB = const_cast<Tensor&>(b);

    mutableA.toCUDA();
    mutableB.toCUDA();
    out.toCUDA();

    int blocks = static_cast<int>(
        (a.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK
        );

    kernels::addKernel<<<blocks, THREADS_PER_BLOCK>>>(
        mutableA.deviceData(),
        mutableB.deviceData(),
        out.deviceData(),
        a.size()
        );

    CUDA_CHECK(cudaGetLastError());
    
}

void launchFlatten3DTo2D(
    const float* input,
    float* output,
    size_t batchSize,
    size_t sequenceLength,
    size_t featureDim
) {
    size_t totalSize = batchSize * sequenceLength * featureDim;

    if (totalSize == 0) {
        return;
    }

    constexpr int threads = 256;
    int blocks = static_cast<int>((totalSize + threads - 1) / threads);

    kernels::flatten3DTo2DKernel << <blocks, threads >> > (
        input,
        output,
        totalSize
        );

    CUDA_CHECK(cudaGetLastError());
    
}

void launchUnflatten2DTo3D(
    const float* input,
    float* output,
    size_t batchSize,
    size_t sequenceLength,
    size_t featureDim
) {
    size_t totalSize = batchSize * sequenceLength * featureDim;

    if (totalSize == 0) {
        return;
    }

    constexpr int threads = 256;
    int blocks = static_cast<int>((totalSize + threads - 1) / threads);

    kernels::unflatten2DTo3DKernel << <blocks, threads >> > (
        input,
        output,
        totalSize
        );

    CUDA_CHECK(cudaGetLastError());
    
}