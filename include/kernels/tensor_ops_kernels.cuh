#pragma once

#include <cstddef>

namespace kernels {

	void launchTensorFill(float* data, size_t size, float value);
	void launchTensorScale(float* data, size_t size, float scalar);
	void launchTensorAdd(const float* a, const float* b, float* out, size_t size);

}