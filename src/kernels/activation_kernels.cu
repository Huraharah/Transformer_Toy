#include "kernels/activation_kernels.cuh"
#include "core/cuda_utils.h"

namespace kernels {
	__global__ void geluForwardKernel(
		float* data,
		size_t size
	) {
		size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) {
			return;
		}
		float x = data[idx];
		data[idx] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
	}

	__global__ void geluBackwardKernel(
		const float* preActivation,
		float* grad,
		size_t size
	) {
		size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) {
			return;
		}
		float x = preActivation[idx];
		float inner = 0.79788456f * (x + 0.044715f * x * x * x);
		float tanhOut = tanhf(inner);

		float innerGrad =
			0.79788456f * (1.0f + 3.0f * 0.044715f * x * x);

		float gradGelu =
			0.5f * (1.0f + tanhOut) +
			0.5f * x * (1.0f - tanhOut * tanhOut) * innerGrad;

		grad[idx] *= gradGelu;
	}
}

void launchGeluForward(float* data, size_t size) {
	if (size == 0) {
		return;
	}

	constexpr int threads = 256;
	int blocks = static_cast<int>((size + threads - 1) / threads);

	kernels::geluForwardKernel << <blocks, threads >> > (data, size);

	CUDA_CHECK(cudaGetLastError());
	cudaSync();
}

void launchGeluBackward(
	const float* preActivation,
	float* grad,
	size_t size
) {
	if (size == 0) {
		return;
	}

	constexpr int threads = 256;
	int blocks = static_cast<int>((size + threads - 1) / threads);

	kernels::geluBackwardKernel << <blocks, threads >> > (
		preActivation,
		grad,
		size
		);

	CUDA_CHECK(cudaGetLastError());
	cudaSync();
}