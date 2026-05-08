#include "layers/attention.h"
#include "core/math_utils.h"

#include <cmath>
#include <vector>
#include <stdexcept>
#include <iostream>

SelfAttention::SelfAttention(size_t embedDim, Random& rng)
    : embedDim_(embedDim),
    queryProj_(embedDim, embedDim, rng),
    keyProj_(embedDim, embedDim, rng),
    valueProj_(embedDim, embedDim, rng),
    outputProj_(embedDim, embedDim, rng) {
}

Tensor SelfAttention::flatten3DTo2D(const Tensor& input) const {
    if (input.rank() != 3) {
        throw std::invalid_argument("flatten3DTo2D expects input shape [batch, sequence, features].");
    }

    size_t batchSize = input.shape()[0];
    size_t sequenceLength = input.shape()[1];
    size_t featureSize = input.shape()[2];

    Tensor output({ batchSize * sequenceLength, featureSize }, 0.0f);

    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t t = 0; t < sequenceLength; ++t) {
            for (size_t f = 0; f < featureSize; ++f) {
                output.at({ b * sequenceLength + t, f }) = input.at({ b, t, f });
            }
        }
    }

    return output;
}

Tensor SelfAttention::unflatten2DTo3D(
    const Tensor& input,
    size_t batchSize,
    size_t sequenceLength
) const {
    if (input.rank() != 2) {
        throw std::invalid_argument("unflatten2DTo3D expects input shape [batch * sequence, features].");
    }

    size_t featureSize = input.shape()[1];

    Tensor output({ batchSize, sequenceLength, featureSize }, 0.0f);

    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t t = 0; t < sequenceLength; ++t) {
            for (size_t f = 0; f < featureSize; ++f) {
                output.at({ b, t, f }) = input.at({ b * sequenceLength + t, f });
            }
        }
    }

    return output;
}

Tensor SelfAttention::forward(const Tensor& input) const {
    if (input.rank() != 3) {
        throw std::invalid_argument("SelfAttention::forward expects input shape [batch, sequence, embedDim].");
    }

    size_t batchSize = input.shape()[0];
    size_t sequenceLength = input.shape()[1];
    size_t inputEmbedDim = input.shape()[2];

    if (inputEmbedDim != embedDim_) {
        throw std::invalid_argument("SelfAttention embed dimension mismatch.");
    }

    Tensor flatInput = flatten3DTo2D(input);

    Tensor qFlat = queryProj_.forward(flatInput);
    Tensor kFlat = keyProj_.forward(flatInput);
    Tensor vFlat = valueProj_.forward(flatInput);

    Tensor Q = unflatten2DTo3D(qFlat, batchSize, sequenceLength);
    Tensor K = unflatten2DTo3D(kFlat, batchSize, sequenceLength);
    Tensor V = unflatten2DTo3D(vFlat, batchSize, sequenceLength);

    Tensor attentionOutput({ batchSize, sequenceLength, embedDim_ }, 0.0f);

    float scale = 1.0f / std::sqrt(static_cast<float>(embedDim_));

    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t t = 0; t < sequenceLength; ++t) {
            std::vector<float> scores(sequenceLength, -1.0e9f);

            for (size_t j = 0; j <= t; ++j) {
                float dot = 0.0f;

                for (size_t f = 0; f < embedDim_; ++f) {
                    dot += Q.at({ b, t, f }) * K.at({ b, j, f });
                }

                scores[j] = dot * scale;
            }

            std::vector<float> weights = MathUtils::softmax(scores);

            /*std::cout << "Token " << t << " weights: ";

            for (size_t j = 0; j < sequenceLength; ++j) {
                std::cout << weights[j] << " ";
            }

            std::cout << "\n";*/

            for (size_t f = 0; f < embedDim_; ++f) {
                float sum = 0.0f;

                for (size_t j = 0; j <= t; ++j) {
                    sum += weights[j] * V.at({ b, j, f });
                }

                attentionOutput.at({ b, t, f }) = sum;
            }
        }
    }

    Tensor flatAttentionOutput = flatten3DTo2D(attentionOutput);
    Tensor projectedFlatOutput = outputProj_.forward(flatAttentionOutput);

    return unflatten2DTo3D(projectedFlatOutput, batchSize, sequenceLength);
}