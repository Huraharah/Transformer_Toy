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

    Tensor flatInput = LayerUtils::flatten3DTo2D(input);

    Tensor qFlat = queryProj_.forward(flatInput);
    Tensor kFlat = keyProj_.forward(flatInput);
    Tensor vFlat = valueProj_.forward(flatInput);

    Tensor Q = LayerUtils::unflatten2DTo3D(qFlat, batchSize, sequenceLength);
    Tensor K = LayerUtils::unflatten2DTo3D(kFlat, batchSize, sequenceLength);
    Tensor V = LayerUtils::unflatten2DTo3D(vFlat, batchSize, sequenceLength);

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

    Tensor flatAttentionOutput = LayerUtils::flatten3DTo2D(attentionOutput);
    Tensor projectedFlatOutput = outputProj_.forward(flatAttentionOutput);

    return LayerUtils::unflatten2DTo3D(projectedFlatOutput, batchSize, sequenceLength);
}