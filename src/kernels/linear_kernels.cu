#include "kernels/linear_kernels.cuh"
#include "kernels/kernel_config.cuh"
#include "core/cuda_utils.h"

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

	__global__ void linearBackwardInputKernel(
		const float* gradOutput,
		const float* weights,
		float* gradInput,
		size_t batchSize,
		size_t inFeatures,
		size_t outFeatures
	) {
		size_t b = blockIdx.y * blockDim.y + threadIdx.y;
		size_t i = blockIdx.x * blockDim.x + threadIdx.x;

		if (b >= batchSize || i >= inFeatures) {
			return;
		}

		float sum = 0.0f;

		for (size_t o = 0; o < outFeatures; ++o) {
			sum += gradOutput[b * outFeatures + o] *
				weights[o * inFeatures + i];
		}

		gradInput[b * inFeatures + i] = sum;
	}

	__global__ void linearBackwardWeightsKernel(
		const float* input,
		const float* gradOutput,
		float* gradWeights,
		size_t batchSize,
		size_t inFeatures,
		size_t outFeatures
	) {
		size_t o = blockIdx.y * blockDim.y + threadIdx.y;
		size_t i = blockIdx.x * blockDim.x + threadIdx.x;

		if (o >= outFeatures || i >= inFeatures) {
			return;
		}

		float sum = 0.0f;

		for (size_t b = 0; b < batchSize; ++b) {
			sum += gradOutput[b * outFeatures + o] *
				input[b * inFeatures + i];
		}

		gradWeights[o * inFeatures + i] = sum;
	}

	__global__ void linearBackwardBiasKernel(
		const float* gradOutput,
		float* gradBias,
		size_t batchSize,
		size_t outFeatures
	) {
		size_t o = blockIdx.x * blockDim.x + threadIdx.x;

		if (o >= outFeatures) {
			return;
		}

		float sum = 0.0f;

		for (size_t b = 0; b < batchSize; ++b) {
			sum += gradOutput[b * outFeatures + o];
		}

		gradBias[o] = sum;
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
		
	}
}

void launchLinearBackward(
	const float* input,
	const float* weights,
	const float* gradOutput,
	float* gradInput,
	float* gradWeights,
	float* gradBias,
	size_t batchSize,
	size_t inFeatures,
	size_t outFeatures
) {
	if (batchSize == 0 || inFeatures == 0 || outFeatures == 0) {
		return;
	}

	constexpr int TILE = 16;

	dim3 block2D(TILE, TILE);

	dim3 gridInput(
		static_cast<unsigned int>((inFeatures + TILE - 1) / TILE),
		static_cast<unsigned int>((batchSize + TILE - 1) / TILE)
	);

	kernels::linearBackwardInputKernel<<<gridInput, block2D>>>(
		gradOutput,
		weights,
		gradInput,
		batchSize,
		inFeatures,
		outFeatures
		);

	CUDA_CHECK(cudaGetLastError());

	dim3 gridWeights(
		static_cast<unsigned int>((inFeatures + TILE - 1) / TILE),
		static_cast<unsigned int>((outFeatures + TILE - 1) / TILE)
	);

	kernels::linearBackwardWeightsKernel<<<gridWeights, block2D>>>(
		input,
		gradOutput,
		gradWeights,
		batchSize,
		inFeatures,
		outFeatures
		);

	CUDA_CHECK(cudaGetLastError());

	int threads = 256;
	int blocks = static_cast<int>((outFeatures + threads - 1) / threads);

	kernels::linearBackwardBiasKernel<<<blocks, threads>>>(
		gradOutput,
		gradBias,
		batchSize,
		outFeatures
		);

	CUDA_CHECK(cudaGetLastError());
	
}