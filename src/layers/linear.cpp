#include "layers/linear.h"
#include "kernels/linear_kernels.cuh"

#include <cmath>
#include <stdexcept>

Linear::Linear(size_t inFeatures, size_t outFeatures, Random& rng, bool useBias)
    : inFeatures_(inFeatures),
    outFeatures_(outFeatures),
    useBias_(useBias),
    weights_(Tensor({ outFeatures, inFeatures }), "linear.weight"),
    bias_(Tensor({ outFeatures }, 0.0f), "linear.bias") {

    float limit = std::sqrt(6.0f / static_cast<float>(inFeatures + outFeatures));

    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_.value[i] = rng.uniform(-limit, limit);
    }
}

Tensor Linear::forward(const Tensor& input) {
    if (input.rank() != 2) {
        throw std::invalid_argument("Linear::forward expects input shape [batchSize, inFeatures].");
    }

    size_t batchSize = input.shape()[0];
    size_t inputFeatures = input.shape()[1];

    if (inputFeatures != inFeatures_) {
        throw std::invalid_argument("Linear::forward input feature size mismatch.");
    }

    Tensor output({ batchSize, outFeatures_ }, 0.0f);
	cachedInput_ = input; // Cache the input for backward pass

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
            outFeatures_,
            useBias_
        );

        return output;
    }

    else {

        for (size_t b = 0; b < batchSize; ++b) {
            for (size_t o = 0; o < outFeatures_; ++o) {
                float sum = useBias_ ? bias_.value[o] : 0.0f;

                for (size_t i = 0; i < inFeatures_; ++i) {
                    sum += input.at({ b, i }) * weights_.value.at({ o, i });
                }

                output.at({ b, o }) = sum;
            }
        }

        return output;
    }
}

Tensor Linear::backward(Tensor& gradOutput)
{
    if (cachedInput_.empty()) {
        throw std::runtime_error("Linear::backward called before forward.");
    }

    if (gradOutput.rank() != 2) {
        throw std::invalid_argument("Linear::backward expects gradOutput shape [batchSize, outFeatures].");
    }

    size_t batchSize = gradOutput.shape()[0];
    size_t gradOutFeatures = gradOutput.shape()[1];

    if (gradOutFeatures != outFeatures_) {
        throw std::invalid_argument("Linear::backward gradOutput feature size mismatch.");
    }

    if (cachedInput_.shape()[0] != batchSize) {
        throw std::invalid_argument("Linear::backward batch size mismatch.");
    }

    if (gradOutput.device() == Device::CUDA || cachedInput_.device() == Device::CUDA) {
        cachedInput_.toCUDA();
        weights_.value.toCUDA();
        gradOutput.toCUDA();

        Tensor gradInput({ batchSize, inFeatures_ }, 0.0f);
        gradInput.toCUDA();

        weights_.grad.fill(0.0f);
        bias_.grad.fill(0.0f);
        weights_.grad.toCUDA();
        bias_.grad.toCUDA();

        launchLinearBackward(
            cachedInput_.deviceData(),
            weights_.value.deviceData(),
            const_cast<Tensor&>(gradOutput).deviceData(),
            gradInput.deviceData(),
            weights_.grad.deviceData(),
            bias_.grad.deviceData(),
            batchSize,
            inFeatures_,
            outFeatures_,
            useBias_
        );

        return gradInput;
    }

    Tensor gradInput({ batchSize, inFeatures_ }, 0.0f);

    weights_.grad.fill(0.0f);
    if (useBias_) {
        bias_.grad.fill(0.0f);
    }

    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t o = 0; o < outFeatures_; ++o) {
            float go = gradOutput.at({ b, o });

            if (useBias_) {
                bias_.grad[o] += go;
            }

            for (size_t i = 0; i < inFeatures_; ++i) {
                weights_.grad.at({ o, i }) += go * cachedInput_.at({ b, i });
                gradInput.at({ b, i }) += go * weights_.value.at({ o, i });
            }
        }
    }

    return gradInput;
}

const Tensor& Linear::weights() const {
    return weights_.value;
}

const Tensor& Linear::bias() const {
    return bias_.value;
}

std::vector<Parameter*> Linear::parameters() {
    if (useBias_) {
        return { &weights_, &bias_ };
    }

    return { &weights_ };
}