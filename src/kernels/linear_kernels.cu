#include "kernels/linear_kernels.cuh"
#include "kernels/kernel_config.cuh"

namespace kernels {

	__global__ void linearForwardKernel(
		const float* input,
		const float* weights,
		const float* bias,
		float* output,
		size_t batchSize,
		size_t inFeatures,
		size_t outFeatures
	) {
		size_t row = blockIdx.y * blockDim.y + threadIdx.y;
		size_t col = blockIdx.x * blockDim.x + threadIdx.x;
		if (row >= batchSize || col >= outFeatures) {
			return;
		}
		float sum = 0.0f;
		for (size_t k = 0; k < inFeatures; ++k) {
			sum += input[row * inFeatures + k] * weights[col * inFeatures + k];
		}
		if (bias != nullptr) {
			sum += bias[col];
		}
		output[row * outFeatures + col] = sum;
	}
}

void launchLinearForward(
	const float* input,
	const float* weights,
	const float* bias,
	float* output,
	size_t batchSize,
	size_t inFeatures,
	size_t outFeatures,
	bool useBias
) {
	if (useBias) {
		constexpr int TILE = 16;

		dim3 block(TILE, TILE);

		dim3 grid(
			(outFeatures + TILE - 1) / TILE,
			(batchSize + TILE - 1) / TILE
		);

		kernels::linearForwardKernel<<<grid, block>>>(input, weights, bias, output, batchSize, inFeatures, outFeatures);

		CUDA_CHECK(cudaGetLastError());
		cudaSync();
	}
}