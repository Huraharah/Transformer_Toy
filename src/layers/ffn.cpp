#include "layers/ffn.h"
#include "core/math_utils.h"
#include "core/parameter.h"

#include <stdexcept>

FFN::FFN(size_t embedDim, size_t hiddenDim, Random& rng)
    : embedDim_(embedDim),
    hiddenDim_(hiddenDim),
    linear1_(embedDim, hiddenDim, rng),
    linear2_(hiddenDim, embedDim, rng) {
}

Tensor FFN::forward(const Tensor& input) const {
    if (input.rank() != 3) {
        throw std::invalid_argument("FFN::forward expects input shape [batch, sequence, embedDim].");
    }

    size_t batchSize = input.shape()[0];
    size_t sequenceLength = input.shape()[1];
    size_t inputEmbedDim = input.shape()[2];

    if (inputEmbedDim != embedDim_) {
        throw std::invalid_argument("FFN embed dimension mismatch.");
    }

    Tensor flatInput({ batchSize * sequenceLength, embedDim_ }, 0.0f);

    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t t = 0; t < sequenceLength; ++t) {
            for (size_t f = 0; f < embedDim_; ++f) {
                flatInput.at({ b * sequenceLength + t, f }) = input.at({ b, t, f });
            }
        }
    }

    Tensor hidden = linear1_.forward(flatInput);

    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = MathUtils::gelu(hidden[i]);
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

std::vector<Parameter*> FFN::parameters() {
    std::vector<Parameter*> params;

    auto p1 = linear1_.parameters();
    auto p2 = linear2_.parameters();

    params.insert(params.end(), p1.begin(), p1.end());
    params.insert(params.end(), p2.begin(), p2.end());

    return params;
}