#include "core/math_utils.h"
#include "kernels/tensor_ops_kernels.cuh"

#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace MathUtils {

    std::vector<float> softmax(const std::vector<float>& values) {
        if (values.empty()) {
            throw std::invalid_argument("softmax expects a non-empty vector.");
        }

        float maxValue = *std::max_element(values.begin(), values.end());

        std::vector<float> expValues(values.size());
        float sumExp = 0.0f;

        for (size_t i = 0; i < values.size(); ++i) {
            expValues[i] = std::exp(values[i] - maxValue);
            sumExp += expValues[i];
        }

        if (sumExp == 0.0f) {
            throw std::runtime_error("softmax sum was zero.");
        }

        for (float& value : expValues) {
            value /= sumExp;
        }

        return expValues;
    }

    float relu(float x) {
        return x > 0.0f ? x : 0.0f;
    }

    float gelu(float x) {
        // Tanh approximation used commonly in Transformer-style networks.
        constexpr float sqrtTwoOverPi = 0.7978845608028654f;
        constexpr float coeff = 0.044715f;

        return 0.5f * x *
            (1.0f + std::tanh(
                sqrtTwoOverPi * (x + coeff * x * x * x)
            ));
    }

    Tensor add(const Tensor& a, const Tensor& b) {
        if (a.shape() != b.shape()) {
            throw std::invalid_argument("Tensor add shape mismatch.");
        }

        Tensor result(a.shape(), 0.0f);

        if (a.device() == Device::CUDA || b.device() == Device::CUDA) {
            launchTensorAdd(
                a, b, result
            );

            return result;
        }

        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] + b[i];
        }

        return result;
    }

    float perplexity(float crossEntropyLoss) {
        return std::exp(crossEntropyLoss);
    }

    float tokenAccuracy(const Tensor& logits, const Tensor& targets) {
        if (logits.rank() != 3) {
            throw std::invalid_argument("tokenAccuracy expects logits shape [batch, sequence, vocab].");
        }

        if (targets.rank() != 2) {
            throw std::invalid_argument("tokenAccuracy expects targets shape [batch, sequence].");
        }

        size_t batchSize = logits.shape()[0];
        size_t sequenceLength = logits.shape()[1];
        size_t vocabSize = logits.shape()[2];

        if (targets.shape()[0] != batchSize || targets.shape()[1] != sequenceLength) {
            throw std::invalid_argument("tokenAccuracy target shape mismatch.");
        }

        size_t correct = 0;
        size_t total = batchSize * sequenceLength;

        for (size_t b = 0; b < batchSize; ++b) {
            for (size_t t = 0; t < sequenceLength; ++t) {
                size_t targetId = static_cast<size_t>(targets.at({ b, t }));

                if (targetId >= vocabSize) {
                    throw std::out_of_range("tokenAccuracy target ID out of range.");
                }

                size_t bestId = 0;
                float bestValue = logits.at({ b, t, 0 });

                for (size_t v = 1; v < vocabSize; ++v) {
                    float value = logits.at({ b, t, v });

                    if (value > bestValue) {
                        bestValue = value;
                        bestId = v;
                    }
                }

                if (bestId == targetId) {
                    ++correct;
                }
            }
        }

        return static_cast<float>(correct) / static_cast<float>(total);
    }

    float geluDerivative(float x) {
        constexpr float sqrt2OverPi = 0.7978845608028654f;
        constexpr float coeff = 0.044715f;

        float x3 = x * x * x;
        float inner = sqrt2OverPi * (x + coeff * x3);
        float tanhInner = std::tanh(inner);

        float sech2 = 1.0f - tanhInner * tanhInner;
        float innerDeriv = sqrt2OverPi * (1.0f + 3.0f * coeff * x * x);

        return 0.5f * (1.0f + tanhInner) + 0.5f * x * sech2 * innerDeriv;
    }
}