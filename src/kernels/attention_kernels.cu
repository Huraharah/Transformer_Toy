#include "kernels/attention_kernels.cuh"
#include "core/cuda_utils.h"

#include <cmath>

namespace kernels {
	__global__ void selfAttentionForwardKernel(
        const float* q,
        const float* k,
        const float* v,
        float* attentionWeights,
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

        float scale = rsqrtf(static_cast<float>(embedDim));

        // Max for causal positions j <= t
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

        // Softmax denominator
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
            attentionWeights[(b * sequenceLength + t) * sequenceLength + j] = weight;
            sumExp += weight;
        }

        // Normalize valid weights and zero masked future positions
        for (size_t j = 0; j < sequenceLength; ++j) {
            size_t idx = (b * sequenceLength + t) * sequenceLength + j;

            if (j <= t) {
                attentionWeights[idx] /= sumExp;
            }
            else {
                attentionWeights[idx] = 0.0f;
            }
        }

        // Weighted value sum
        for (size_t f = 0; f < embedDim; ++f) {
            float sum = 0.0f;

            for (size_t j = 0; j <= t; ++j) {
                float weight =
                    attentionWeights[(b * sequenceLength + t) * sequenceLength + j];

                sum += weight *
                    v[(b * sequenceLength + j) * embedDim + f];
            }

            output[(b * sequenceLength + t) * embedDim + f] = sum;
        }
    }

	__global__ void selfAttentionBackwardKernel(
		const float* q,
		const float* k,
		const float* v,
		const float* attentionWeights,
		const float* gradOutput,
		float* gradQ,
		float* gradK,
		float* gradV,
		size_t batchSize,
		size_t sequenceLength,
		size_t embedDim
	) {
        {
            size_t b = blockIdx.x;
            size_t t = blockIdx.y;

            if (b >= batchSize || t >= sequenceLength) {
                return;
            }

            float scale = rsqrtf(static_cast<float>(embedDim));

            for (size_t j = 0; j <= t; ++j) {
                float gradWeight = 0.0f;

                for (size_t f = 0; f < embedDim; ++f) {
                    gradWeight +=
                        gradOutput[(b * sequenceLength + t) * embedDim + f] *
                        v[(b * sequenceLength + j) * embedDim + f];

                    atomicAdd(
                        &gradV[(b * sequenceLength + j) * embedDim + f],
                        attentionWeights[(b * sequenceLength + t) * sequenceLength + j] *
                        gradOutput[(b * sequenceLength + t) * embedDim + f]
                    );
                }

                float weightedSum = 0.0f;

                for (size_t m = 0; m <= t; ++m) {
                    float gw = 0.0f;

                    for (size_t f = 0; f < embedDim; ++f) {
                        gw +=
                            gradOutput[(b * sequenceLength + t) * embedDim + f] *
                            v[(b * sequenceLength + m) * embedDim + f];
                    }

                    weightedSum +=
                        gw *
                        attentionWeights[(b * sequenceLength + t) * sequenceLength + m];
                }

                float gradScore =
                    attentionWeights[(b * sequenceLength + t) * sequenceLength + j] *
                    (gradWeight - weightedSum);

                gradScore *= scale;

                for (size_t f = 0; f < embedDim; ++f) {
                    atomicAdd(
                        &gradQ[(b * sequenceLength + t) * embedDim + f],
                        gradScore * k[(b * sequenceLength + j) * embedDim + f]
                    );

                    atomicAdd(
                        &gradK[(b * sequenceLength + j) * embedDim + f],
                        gradScore * q[(b * sequenceLength + t) * embedDim + f]
                    );
                }
            }
        }

	}

    __global__ void multiHeadAttentionForwardKernel(
        const float* q,
        const float* k,
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

	__global__ void multiHeadAttentionBackwardKernel(
		const float* q,
		const float* k,
		const float* v,
		const float* attentionWeights,
		const float* gradOutput,
		float* gradQ,
		float* gradK,
		float* gradV,
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

        for (size_t j = 0; j <= t; ++j) {
            float gradWeight = 0.0f;

            for (size_t f = 0; f < headDim; ++f) {
                size_t globalF = h * headDim + f;

                gradWeight +=
                    gradOutput[(b * sequenceLength + t) * embedDim + globalF] *
                    v[(b * sequenceLength + j) * embedDim + globalF];

                atomicAdd(
                    &gradV[(b * sequenceLength + j) * embedDim + globalF],
                    attentionWeights[
                        ((b * numHeads + h) * sequenceLength + t) * sequenceLength + j
                    ] *
                    gradOutput[(b * sequenceLength + t) * embedDim + globalF]
                            );
            }

            float weightedSum = 0.0f;

            for (size_t m = 0; m <= t; ++m) {
                float gw = 0.0f;

                for (size_t f = 0; f < headDim; ++f) {
                    size_t globalF = h * headDim + f;

                    gw +=
                        gradOutput[(b * sequenceLength + t) * embedDim + globalF] *
                        v[(b * sequenceLength + m) * embedDim + globalF];
                }

                weightedSum +=
                    gw *
                    attentionWeights[
                        ((b * numHeads + h) * sequenceLength + t) * sequenceLength + m
                    ];
            }

            float gradScore =
                attentionWeights[
                    ((b * numHeads + h) * sequenceLength + t) * sequenceLength + j
                ] *
                (gradWeight - weightedSum);

                    gradScore *= scale;

                    for (size_t f = 0; f < headDim; ++f) {
                        size_t globalF = h * headDim + f;

                        atomicAdd(
                            &gradQ[(b * sequenceLength + t) * embedDim + globalF],
                            gradScore *
                            k[(b * sequenceLength + j) * embedDim + globalF]
                        );

                        atomicAdd(
                            &gradK[(b * sequenceLength + j) * embedDim + globalF],
                            gradScore *
                            q[(b * sequenceLength + t) * embedDim + globalF]
                        );
                    }
        }
    }

}

void launchSelfAttentionForward(
    const float* q,
    const float* k,
    const float* v,
    float* attentionWeights,
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

    kernels::selfAttentionForwardKernel<<<grid, block>>>(
        q,
        k,
        v,
        attentionWeights,
        output,
        batchSize,
        sequenceLength,
        embedDim
        );

    CUDA_CHECK(cudaGetLastError());
    cudaSync();
}

void launchSelfAttentionBackward(
	const float* q,
	const float* k,
	const float* v,
	const float* attentionWeights,
	const float* gradOutput,
	float* gradQ,
	float* gradK,
	float* gradV,
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

    kernels::selfAttentionBackwardKernel<<<grid, block>>>(
        q,
        k,
        v,
        attentionWeights,
        gradOutput,
        gradQ,
        gradK,
        gradV,
        batchSize,
        sequenceLength,
        embedDim
        );

    CUDA_CHECK(cudaGetLastError());
    cudaSync();
}

void launchMultiHeadAttentionForward(
    const float* q,
    const float* k,
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

    kernels::multiHeadAttentionForwardKernel<<<grid, block>>>(
        q,
        k,
        v,
        attentionWeights,
        output,
        batchSize,
        sequenceLength,
        numHeads,
        headDim
        );

    CUDA_CHECK(cudaGetLastError());
    cudaSync();
}

void launchMultiHeadAttentionBackward(
	const float* q,
	const float* k,
	const float* v,
	const float* attentionWeights,
	const float* gradOutput,
	float* gradQ,
	float* gradK,
	float* gradV,
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

    kernels::multiHeadAttentionBackwardKernel<<<grid, block>>>(
        q,
        k,
        v,
        attentionWeights,
        gradOutput,
        gradQ,
        gradK,
        gradV,
        batchSize,
        sequenceLength,
        numHeads,
        headDim
        );

    CUDA_CHECK(cudaGetLastError());
    cudaSync();
}