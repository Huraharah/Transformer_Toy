#include "layers/layer_norm.h"

#include <cmath>
#include <stdexcept>

LayerNorm::LayerNorm(size_t featureDim, float epsilon)
    : featureDim_(featureDim),
    epsilon_(epsilon),
    gamma_({ featureDim }, 1.0f),
    beta_({ featureDim }, 0.0f) {
}

Tensor LayerNorm::forward(const Tensor& input) const {
    if (input.rank() != 3) {
        throw std::invalid_argument(
            "LayerNorm::forward expects input shape [batch, sequence, features]."
        );
    }

    size_t batchSize = input.shape()[0];
    size_t sequenceLength = input.shape()[1];
    size_t featureSize = input.shape()[2];

    if (featureSize != featureDim_) {
        throw std::invalid_argument(
            "LayerNorm feature dimension mismatch."
        );
    }

    Tensor output(input.shape(), 0.0f);

    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t t = 0; t < sequenceLength; ++t) {

            // Compute mean
            float mean = 0.0f;

            for (size_t f = 0; f < featureSize; ++f) {
                mean += input.at({ b, t, f });
            }

            mean /= static_cast<float>(featureSize);

            // Compute variance
            float variance = 0.0f;

            for (size_t f = 0; f < featureSize; ++f) {
                float diff = input.at({ b, t, f }) - mean;
                variance += diff * diff;
            }

            variance /= static_cast<float>(featureSize);

            float invStdDev = 1.0f / std::sqrt(variance + epsilon_);

            // Normalize
            for (size_t f = 0; f < featureSize; ++f) {
                float normalized =
                    (input.at({ b, t, f }) - mean) * invStdDev;

                output.at({ b, t, f }) =
                    normalized * gamma_.at({ f }) +
                    beta_.at({ f });
            }
        }
    }

    return output;
}

const Tensor& LayerNorm::gamma() const {
    return gamma_;
}

const Tensor& LayerNorm::beta() const {
    return beta_;
}