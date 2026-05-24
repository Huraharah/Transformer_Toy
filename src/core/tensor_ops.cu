#include "core/tensor_ops.h"
#include "core/cuda_utils.h"

#include <stdexcept>

static constexpr int THREADS_PER_BLOCK = 256;

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

void tensorFillCUDA(Tensor& tensor, float value) {
    if (tensor.empty()) {
        return;
    }

    tensor.toCUDA();

    int blocks = static_cast<int>(
        (tensor.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK
        );

    fillKernel <<<blocks, THREADS_PER_BLOCK >>> (
        tensor.deviceData(),
        tensor.size(),
        value
        );

    CUDA_CHECK(cudaGetLastError());
    cudaSync();
}

void tensorScaleCUDA(Tensor& tensor, float scalar) {
    if (tensor.empty()) {
        return;
    }

    tensor.toCUDA();

    int blocks = static_cast<int>(
        (tensor.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK
        );

    scaleKernel <<<blocks, THREADS_PER_BLOCK >>> (
        tensor.deviceData(),
        tensor.size(),
        scalar
        );

    CUDA_CHECK(cudaGetLastError());
    cudaSync();
}

void tensorAddCUDA(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.size() != b.size() || a.size() != out.size()) {
        throw std::invalid_argument("tensorAddCUDA requires tensors with matching sizes.");
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

    addKernel <<<blocks, THREADS_PER_BLOCK >>> (
        mutableA.deviceData(),
        mutableB.deviceData(),
        out.deviceData(),
        a.size()
        );

    CUDA_CHECK(cudaGetLastError());
    cudaSync();
}