#pragma once

#include "core/tensor.h"

#include <vector>

namespace MathUtils {

	std::vector<float> softmax(const std::vector<float>& values);

	float relu(float x);

	float gelu(float x);

	Tensor add(const Tensor& a, const Tensor& b);

	float crossEntropyLoss(const Tensor& logits, const Tensor& targets);

	float perplexity(float crossEntropyLoss);

	float tokenAccuracy(const Tensor& logits, const Tensor& targets);

}