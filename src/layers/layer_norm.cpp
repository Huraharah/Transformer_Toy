#include "layers/layer_norm.h"

#include <cmath>
#include <stdexcept>

LayerNorm::LayerNorm(size_t featureDim, float epsilon)
    : featureDim_(featureDim),
    epsilon_(epsilon),
    gamma_(Tensor({ featureDim }, 1.0f), "layernorm.gamma"),
    beta_(Tensor({ featureDim }, 0.0f), "layernorm.beta") {
}

Tensor LayerNorm::forward(const Tensor& input) {
    if (input.rank() != 3) {
        throw std::invalid_argument(
            "LayerNorm::forward expects input shape [batch, sequence, features]."
        );
    }

	cachedInput_ = input;

    size_t batchSize = input.shape()[0];
    size_t sequenceLength = input.shape()[1];
    size_t featureSize = input.shape()[2];

    cachedMean_ = Tensor({ batchSize, sequenceLength }, 0.0f);
    cachedInvStd_ = Tensor({ batchSize, sequenceLength }, 0.0f);

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

			cachedMean_.at({ b, t }) = mean;

            // Compute variance
            float variance = 0.0f;

            for (size_t f = 0; f < featureSize; ++f) {
                float diff = input.at({ b, t, f }) - mean;
                variance += diff * diff;
            }

            variance /= static_cast<float>(featureSize);

            float invStdDev = 1.0f / std::sqrt(variance + epsilon_);

			cachedInvStd_.at({ b, t }) = invStdDev;

            // Normalize
            for (size_t f = 0; f < featureSize; ++f) {
                float normalized =
                    (input.at({ b, t, f }) - mean) * invStdDev;

                output.at({ b, t, f }) =
                    normalized * gamma_.value.at({ f }) +
                    beta_.value.at({ f });
            }
        }
    }
    return output;
}

Tensor LayerNorm::backward(const Tensor& gradOutput) {
    if (cachedInput_.empty()) {
        throw std::runtime_error("LayerNorm::backward called before forward.");
    }

    if (gradOutput.rank() != 3) {
        throw std::invalid_argument("LayerNorm::backward expects gradOutput shape [batch, sequence, featureDim].");
    }

    size_t batchSize = gradOutput.shape()[0];
    size_t sequenceLength = gradOutput.shape()[1];
    size_t gradFeatureDim = gradOutput.shape()[2];

    if (gradFeatureDim != featureDim_) {
        throw std::invalid_argument("LayerNorm::backward feature dimension mismatch.");
    }

    if (cachedInput_.shape() != gradOutput.shape()) {
        throw std::invalid_argument("LayerNorm::backward cached input shape mismatch.");
    }

    Tensor gradInput(gradOutput.shape(), 0.0f);

    gamma_.grad.fill(0.0f);
    beta_.grad.fill(0.0f);

    float invN = 1.0f / static_cast<float>(featureDim_);

    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t t = 0; t < sequenceLength; ++t) {
            float mean = cachedMean_.at({ b, t });
            float invStd = cachedInvStd_.at({ b, t });

            float sumDyGamma = 0.0f;
            float sumDyGammaXhat = 0.0f;

            for (size_t f = 0; f < featureDim_; ++f) {
                float x = cachedInput_.at({ b, t, f });
                float xHat = (x - mean) * invStd;

                float dy = gradOutput.at({ b, t, f });
                float dyGamma = dy * gamma_.value[f];

                sumDyGamma += dyGamma;
                sumDyGammaXhat += dyGamma * xHat;

                gamma_.grad[f] += dy * xHat;
                beta_.grad[f] += dy;
            }

            for (size_t f = 0; f < featureDim_; ++f) {
                float x = cachedInput_.at({ b, t, f });
                float xHat = (x - mean) * invStd;

                float dy = gradOutput.at({ b, t, f });
                float dyGamma = dy * gamma_.value[f];

                gradInput.at({ b, t, f }) =
                    invN * invStd *
                    (
                        static_cast<float>(featureDim_) * dyGamma
                        - sumDyGamma
                        - xHat * sumDyGammaXhat
                        );
            }
        }
    }

    return gradInput;
}

std::vector<Parameter*> LayerNorm::parameters() {
    return { &gamma_, &beta_ };
}