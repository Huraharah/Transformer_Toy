#include "layers/attention.h"
#include "core/math_utils.h"
#include "core/parameter.h"

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

std::vector<Parameter*> SelfAttention::parameters() {
    std::vector<Parameter*> params;

    auto append = [&params](std::vector<Parameter*> more) {
        params.insert(params.end(), more.begin(), more.end());
        };

    append(queryProj_.parameters());
    append(keyProj_.parameters());
    append(valueProj_.parameters());
    append(outputProj_.parameters());

    return params;
}

MultiHeadAttention::MultiHeadAttention(const AttentionConfig& config, Random& rng)
    : config_(config),
    embedDim_(config.embedDim),
    numHeads_(config.numHeads),
    headDim_(config.embedDim / config.numHeads),
    queryProj_(config.embedDim, config.embedDim, rng),
    keyProj_(config.embedDim, config.embedDim, rng),
    valueProj_(config.embedDim, config.embedDim, rng),
    outputProj_(config.embedDim, config.embedDim, rng) {
}

Tensor MultiHeadAttention::forward(const Tensor& input) const {
    if (input.rank() != 3) {
        throw std::invalid_argument("MultiHeadAttention::forward expects input shape [batch, sequence, embedDim].");
    }

    size_t batchSize = input.shape()[0];
    size_t sequenceLength = input.shape()[1];
    size_t inputEmbedDim = input.shape()[2];

    if (inputEmbedDim != embedDim_) {
        throw std::invalid_argument("MultiHeadAttention embed dimension mismatch.");
    }

    Tensor flatInput = LayerUtils::flatten3DTo2D(input);

    Tensor qFlat = queryProj_.forward(flatInput);
    Tensor kFlat = keyProj_.forward(flatInput);
    Tensor vFlat = valueProj_.forward(flatInput);

    Tensor Q = LayerUtils::unflatten2DTo3D(qFlat, batchSize, sequenceLength);
    Tensor K = LayerUtils::unflatten2DTo3D(kFlat, batchSize, sequenceLength);
    Tensor V = LayerUtils::unflatten2DTo3D(vFlat, batchSize, sequenceLength);

    Tensor attentionOutput({ batchSize, sequenceLength, embedDim_ }, 0.0f);

    float scale = 1.0f / std::sqrt(static_cast<float>(headDim_));

    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t h = 0; h < numHeads_; ++h) {
            for (size_t t = 0; t < sequenceLength; ++t) {
                std::vector<float> scores(sequenceLength, -1.0e9f);

                for (size_t j = 0; j <= t; ++j) {
                    float dot = 0.0f;

                    for (size_t f = 0; f < headDim_; ++f) {
                        size_t idx = h * headDim_ + f;
                        dot += Q.at({ b, t, idx }) * K.at({ b, j, idx });
                    }

                    scores[j] = dot * scale;
                }

                std::vector<float> weights = MathUtils::softmax(scores);

                for (size_t f = 0; f < headDim_; ++f) {
                    size_t idx = h * headDim_ + f;
                    float sum = 0.0f;

                    for (size_t j = 0; j <= t; ++j) {
                        sum += weights[j] * V.at({ b, j, idx });
                    }

                    attentionOutput.at({ b, t, idx }) = sum;
                }
            }
        }
    }

    Tensor flatAttentionOutput = LayerUtils::flatten3DTo2D(attentionOutput);
    Tensor projectedFlatOutput = outputProj_.forward(flatAttentionOutput);

    return LayerUtils::unflatten2DTo3D(projectedFlatOutput, batchSize, sequenceLength);
}

std::vector<Parameter*> MultiHeadAttention::parameters() {
    std::vector<Parameter*> params;

    auto append = [&params](std::vector<Parameter*> more) {
        params.insert(params.end(), more.begin(), more.end());
        };

    append(queryProj_.parameters());
    append(keyProj_.parameters());
    append(valueProj_.parameters());
    append(outputProj_.parameters());

    return params;
}