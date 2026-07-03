#include "layers/linear.h"
#include "kernels/linear_kernels.cuh"

#include <cmath>
#include <stdexcept>

Linear::Linear(size_t inFeatures, size_t outFeatures, Random& rng)
    : inFeatures_(inFeatures),
    outFeatures_(outFeatures),
    weights_(Tensor({ outFeatures, inFeatures }), "linear.weight"),
    bias_(Tensor({ outFeatures }, 0.0f), "linear.bias") {

    float limit = std::sqrt(6.0f / static_cast<float>(inFeatures + outFeatures));

    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_.value[i] = rng.uniform(-limit, limit);
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

    if (input.device() == Device::CUDA) {
        Tensor& mutableInput = const_cast<Tensor&>(input);
        Tensor& mutableWeights = const_cast<Tensor&>(weights_.value);
        Tensor& mutableBias = const_cast<Tensor&>(bias_.value);

        mutableInput.toCUDA();
        mutableWeights.toCUDA();
        mutableBias.toCUDA();
        output.toCUDA();

        launchLinearForward(
            mutableInput.deviceData(),
            mutableWeights.deviceData(),
            mutableBias.deviceData(),
            output.deviceData(),
            batchSize,
            inFeatures_,
            outFeatures_
        );

        return output;
    }

    else {

        for (size_t b = 0; b < batchSize; ++b) {
            for (size_t o = 0; o < outFeatures_; ++o) {
                float sum = bias_.value[o];

                for (size_t i = 0; i < inFeatures_; ++i) {
                    sum += input.at({ b, i }) * weights_.value.at({ o, i });
                }

                output.at({ b, o }) = sum;
            }
        }

        return output;
    }
}

const Tensor& Linear::weights() const {
    return weights_.value;
}

const Tensor& Linear::bias() const {
    return bias_.value;
}

std::vector<Parameter*> Linear::parameters() {
    return { &weights_, &bias_ };
}