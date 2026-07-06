#include "layers/ffn.h"
#include "core/math_utils.h"
#include "core/parameter.h"
#include "kernels/activation_kernels.cuh"

#include <stdexcept>

FFN::FFN(size_t embedDim, size_t hiddenDim, Random& rng)
    : embedDim_(embedDim),
    hiddenDim_(hiddenDim),
    linear1_(embedDim, hiddenDim, rng),
    linear2_(hiddenDim, embedDim, rng) {
}

Tensor FFN::forward(const Tensor& input){
    if (input.rank() != 3) {
        throw std::invalid_argument("FFN::forward expects input shape [batch, sequence, embedDim].");
    }

    size_t batchSize = input.shape()[0];
    size_t sequenceLength = input.shape()[1];
    size_t inputEmbedDim = input.shape()[2];

    if (inputEmbedDim != embedDim_) {
        throw std::invalid_argument("FFN embed dimension mismatch.");
    }

	cachedInputShape_ = { batchSize, sequenceLength, embedDim_ }; // Cache the input shape for backward pass

    Tensor flatInput({ batchSize * sequenceLength, embedDim_ }, 0.0f);

    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t t = 0; t < sequenceLength; ++t) {
            for (size_t f = 0; f < embedDim_; ++f) {
                flatInput.at({ b * sequenceLength + t, f }) = input.at({ b, t, f });
            }
        }
    }

    Tensor hidden = linear1_.forward(flatInput);

	cachedHiddenPreActivation_ = hidden; // Cache for backward pass

	if (hidden.device() == Device::CUDA) {
        launchGeluForward(hidden.deviceData(), hidden.size());
    } else {
		for (size_t i = 0; i < hidden.size(); ++i) {
			hidden[i] = MathUtils::gelu(hidden[i]);
		}
    }

    Tensor outputFlat = linear2_.forward(hidden);

    Tensor output({ batchSize, sequenceLength, embedDim_ }, 0.0f);

    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t t = 0; t < sequenceLength; ++t) {
            for (size_t f = 0; f < embedDim_; ++f) {
                output.at({ b, t, f }) = outputFlat.at({ b * sequenceLength + t, f });
            }
        }
    }

    return output;
}

Tensor FFN::backward(const Tensor& gradOutput) {
	if (gradOutput.rank() != 3) {
		throw std::invalid_argument("FFN::backward expects gradOutput shape [batch, sequence, embedDim].");
	}
	size_t batchSize = gradOutput.shape()[0];
	size_t sequenceLength = gradOutput.shape()[1];
	size_t outputEmbedDim = gradOutput.shape()[2];
	if (outputEmbedDim != embedDim_) {
		throw std::invalid_argument("FFN backward embed dimension mismatch.");
	}
	if (cachedInputShape_.empty()) {
		throw std::runtime_error("FFN::backward called before forward.");
	}
	if (cachedInputShape_[0] != batchSize || cachedInputShape_[1] != sequenceLength) {
		throw std::invalid_argument("FFN backward batch or sequence size mismatch.");
	}

	Tensor flatGradOutput({ batchSize * sequenceLength, embedDim_ }, 0.0f);
	for (size_t b = 0; b < batchSize; ++b) {
		for (size_t t = 0; t < sequenceLength; ++t) {
			for (size_t f = 0; f < embedDim_; ++f) {
				flatGradOutput.at({ b * sequenceLength + t, f }) = gradOutput.at({ b, t, f });
			}
		}
	}

	Tensor gradHidden = linear2_.backward(flatGradOutput);

	if (gradHidden.device() == Device::CUDA) {
		launchGeluBackward(cachedHiddenPreActivation_.deviceData(), gradHidden.deviceData(), gradHidden.size());
	}
	else {
		for (size_t i = 0; i < gradHidden.size(); ++i) {
			float x = cachedHiddenPreActivation_[i];
			float geluGrad = MathUtils::geluDerivative(x);
			gradHidden[i] *= geluGrad;
		}
	}

	if (cachedHiddenPreActivation_.size() != gradHidden.size()) {
		throw std::runtime_error("FFN backward cached activation size mismatch.");
	}

	Tensor gradInputFlat = linear1_.backward(gradHidden);
	Tensor gradInput({ batchSize, sequenceLength, embedDim_ }, 0.0f);
	for (size_t b = 0; b < batchSize; ++b) {
		for (size_t t = 0; t < sequenceLength; ++t) {
			for (size_t f = 0; f < embedDim_; ++f) {
				gradInput.at({ b, t, f }) = gradInputFlat.at({ b * sequenceLength + t, f });
			}
		}
	}
	return gradInput;
}

std::vector<Parameter*> FFN::parameters() {
    std::vector<Parameter*> params;

    auto p1 = linear1_.parameters();
    auto p2 = linear2_.parameters();

    params.insert(params.end(), p1.begin(), p1.end());
    params.insert(params.end(), p2.begin(), p2.end());

    return params;
}