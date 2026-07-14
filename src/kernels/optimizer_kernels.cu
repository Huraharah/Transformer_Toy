#include "kernels/optimizer_kernels.cuh"
#include "core/cuda_utils.h"

namespace kernels{

	__global__ void adamUpdateKernel(
		float* values,
		const float* grads,
		float* m,
		float* v,
		size_t size,
		float learningRate,
		float beta1,
		float beta2,
		float epsilon,
		float weightDecay,
		int timestep
	) {
		size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < size) {
			float grad = grads[idx];
			if (weightDecay != 0.0f) {
				grad += weightDecay * values[idx];
			}
			m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;
			v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
			float mHat = m[idx] / (1.0f - powf(beta1, timestep));
			float vHat = v[idx] / (1.0f - powf(beta2, timestep));
			values[idx] -= learningRate * mHat / (sqrtf(vHat) + epsilon);
		}
	}


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

void launchAdamUpdateKernel(
    float* values,
    const float* grads,
    float* m,
    float* v,
    size_t size,
    float learningRate,
    float beta1,
    float beta2,
    float epsilon,
    float weightDecay,
    int timestep
) {
	if (size == 0) {
		return;
	}
	int blocks = getBlockCount(size, THREADS_PER_BLOCK);
	kernels::adamUpdateKernel<<<blocks, THREADS_PER_BLOCK>>>(
		values,
		grads,
		m,
		v,
		size,
		learningRate,
		beta1,
		beta2,
		epsilon,
		weightDecay,
		timestep
		);
	CUDA_CHECK(cudaGetLastError());
	
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

	kernels::sgdUpdateKernel<<<blocks, THREADS_PER_BLOCK>>>(
		values,
		grads,
		size,
		learningRate,
		weightDecay
		);

	CUDA_CHECK(cudaGetLastError());
	
}