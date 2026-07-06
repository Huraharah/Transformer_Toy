#pragma once
#include "core/cuda_utils.h"
#include "core/tensor.h"

#include <cstddef>


	void launchTensorFill(Tensor& tensor, float value);
	void launchTensorScale(Tensor& tensor, float scalar);
	void launchTensorAdd(const Tensor& a, const Tensor& b, Tensor& out);

    void launchFlatten3DTo2D(
        const float* input,
        float* output,
        size_t batchSize,
        size_t sequenceLength,
        size_t featureDim
    );

    void launchUnflatten2DTo3D(
        const float* input,
        float* output,
        size_t batchSize,
        size_t sequenceLength,
        size_t featureDim
    );