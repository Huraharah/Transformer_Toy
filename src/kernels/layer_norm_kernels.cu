#include "kernels/layer_norm_kernels.cuh"
#include "core/cuda_utils.h"

#include <stdexcept>

namespace kernels {

    __global__ void layerNormForwardKernel(
        const float* input,
        const float* gamma,
        const float* beta,
        float* output,
        float* means,
        float* invStds,
        size_t batchSize,
        size_t sequenceLength,
        size_t featureDim,
        float epsilon
    ) {
        size_t row = blockIdx.x;
        size_t totalRows = batchSize * sequenceLength;

        if (row >= totalRows) {
            return;
        }

        size_t base = row * featureDim;

        float mean = 0.0f;

        for (size_t f = 0; f < featureDim; ++f) {
            mean += input[base + f];
        }

        mean /= static_cast<float>(featureDim);

        float variance = 0.0f;

        for (size_t f = 0; f < featureDim; ++f) {
            float diff = input[base + f] - mean;
            variance += diff * diff;
        }

        variance /= static_cast<float>(featureDim);

        float invStd = rsqrtf(variance + epsilon);

        means[row] = mean;
        invStds[row] = invStd;

        for (size_t f = 0; f < featureDim; ++f) {
            float xHat = (input[base + f] - mean) * invStd;
            output[base + f] = gamma[f] * xHat + beta[f];
        }
    }

    __global__ void layerNormBackwardKernel(
        const float* input,
        const float* gradOutput,
        const float* gamma,
        const float* means,
        const float* invStds,
        float* gradInput,
        float* gradGamma,
        float* gradBeta,
        size_t batchSize,
        size_t sequenceLength,
        size_t featureDim
    ) {
        size_t row = blockIdx.x;
        size_t totalRows = batchSize * sequenceLength;

        if (row >= totalRows) {
            return;
        }

        size_t base = row * featureDim;

        float mean = means[row];
        float invStd = invStds[row];
        float invN = 1.0f / static_cast<float>(featureDim);

        float sumDyGamma = 0.0f;
        float sumDyGammaXhat = 0.0f;

        for (size_t f = 0; f < featureDim; ++f) {
            float xHat = (input[base + f] - mean) * invStd;
            float dy = gradOutput[base + f];
            float dyGamma = dy * gamma[f];

            sumDyGamma += dyGamma;
            sumDyGammaXhat += dyGamma * xHat;

            atomicAdd(&gradGamma[f], dy * xHat);
            atomicAdd(&gradBeta[f], dy);
        }

        for (size_t f = 0; f < featureDim; ++f) {
            float xHat = (input[base + f] - mean) * invStd;
            float dy = gradOutput[base + f];
            float dyGamma = dy * gamma[f];

            gradInput[base + f] =
                invN * invStd *
                (
                    static_cast<float>(featureDim) * dyGamma
                    - sumDyGamma
                    - xHat * sumDyGammaXhat
                    );
        }
    }

}

void launchLayerNormForward(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    float* means,
    float* invStds,
    size_t batchSize,
    size_t sequenceLength,
    size_t featureDim,
    float epsilon
) {
    if (batchSize == 0 || sequenceLength == 0 || featureDim == 0) {
        return;
    }

    int blocks = static_cast<int>(batchSize * sequenceLength);
    int threads = 1;

    kernels::layerNormForwardKernel<<<blocks, threads>>>(
        input,
        gamma,
        beta,
        output,
        means,
        invStds,
        batchSize,
        sequenceLength,
        featureDim,
        epsilon
        );

    CUDA_CHECK(cudaGetLastError());
    
}

void launchLayerNormBackward(
    const float* input,
    const float* gradOutput,
    const float* gamma,
    const float* means,
    const float* invStds,
    float* gradInput,
    float* gradGamma,
    float* gradBeta,
    size_t batchSize,
    size_t sequenceLength,
    size_t featureDim
) {
    if (batchSize == 0 || sequenceLength == 0 || featureDim == 0) {
        return;
    }

    int blocks = static_cast<int>(batchSize * sequenceLength);
    int threads = 1;

    kernels::layerNormBackwardKernel<<<blocks, threads>>>(
        input,
        gradOutput,
        gamma,
        means,
        invStds,
        gradInput,
        gradGamma,
        gradBeta,
        batchSize,
        sequenceLength,
        featureDim
        );

    CUDA_CHECK(cudaGetLastError());
    
}