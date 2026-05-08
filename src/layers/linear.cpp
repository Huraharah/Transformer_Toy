#include "layers/linear.h"

#include <cmath>
#include <stdexcept>

Linear::Linear(size_t inFeatures, size_t outFeatures, Random& rng)
    : inFeatures_(inFeatures),
    outFeatures_(outFeatures),
    weights_({ outFeatures, inFeatures }),
    bias_({ outFeatures }, 0.0f) {

    float limit = std::sqrt(6.0f / static_cast<float>(inFeatures + outFeatures));

    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] = rng.uniform(-limit, limit);
    }
}

Tensor Linear::forward(const Tensor& input) const {
    if (input.rank() != 2) {
        throw std::invalid_argument("Linear::forward expects input shape [batchSize, inFeatures].");
    }

    size_t batchSize = input.shape()[0];
    size_t inputFeatures = input.shape()[1];

    if (inputFeatures != inFeatures_) {
        throw std::invalid_argument("Linear::forward input feature size mismatch.");
    }

    Tensor output({ batchSize, outFeatures_ }, 0.0f);

    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t o = 0; o < outFeatures_; ++o) {
            float sum = bias_[o];

            for (size_t i = 0; i < inFeatures_; ++i) {
                sum += input.at({ b, i }) * weights_.at({ o, i });
            }

            output.at({ b, o }) = sum;
        }
    }

    return output;
}

const Tensor& Linear::weights() const {
    return weights_;
}

const Tensor& Linear::bias() const {
    return bias_;
}