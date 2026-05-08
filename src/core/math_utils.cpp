#include "core/math_utils.h"

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

    Tensor MathUtils::add(const Tensor& a, const Tensor& b) {
        if (a.shape() != b.shape()) {
            throw std::invalid_argument("Tensor add shape mismatch.");
        }

        Tensor output(a.shape(), 0.0f);

        for (size_t i = 0; i < a.size(); ++i) {
            output[i] = a[i] + b[i];
        }

        return output;
    }
}