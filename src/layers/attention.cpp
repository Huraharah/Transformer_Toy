#include "layers/attention.h"
#include "core/math_utils.h"
#include "core/parameter.h"
#include "kernels/attention_kernels.cuh"

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

Tensor SelfAttention::forward(const Tensor& input){
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

    cachedInput_ = input;
    cachedQ_ = Q;
    cachedK_ = K;
    cachedV_ = V;
    cachedAttentionWeights_ = Tensor({ batchSize, sequenceLength, sequenceLength }, 0.0f);

    if (input.device() == Device::CUDA) {
        Q.toCUDA();
        K.toCUDA();
        V.toCUDA();
        cachedAttentionWeights_.toCUDA();

        Tensor attended({ batchSize, sequenceLength, embedDim_ }, 0.0f);
        attended.toCUDA();

        launchSelfAttentionForward(
            cachedQ_.deviceData(),
            cachedK_.deviceData(),
            cachedV_.deviceData(),
            cachedAttentionWeights_.deviceData(),
            attended.deviceData(),
            batchSize,
            sequenceLength,
            embedDim_
        );

        Tensor flatAttended = LayerUtils::flatten3DTo2D(attended);
        Tensor projectedFlat = outputProj_.forward(flatAttended);

        return LayerUtils::unflatten2DTo3D(
            projectedFlat,
            batchSize,
            sequenceLength
        );
    }

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

            for (size_t j = 0; j < sequenceLength; ++j) {
                cachedAttentionWeights_.at({ b, t, j }) = weights[j];
            }

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

Tensor SelfAttention::backward(const Tensor& gradOutput) {
    if (cachedInput_.empty()) {
        throw std::runtime_error("SelfAttention::backward called before forward.");
    }

    size_t batchSize = gradOutput.shape()[0];
    size_t sequenceLength = gradOutput.shape()[1];

    Tensor flatGradOutput = LayerUtils::flatten3DTo2D(gradOutput);
    Tensor gradAttentionFlat = outputProj_.backward(flatGradOutput);
    Tensor gradAttention = LayerUtils::unflatten2DTo3D(
        gradAttentionFlat,
        batchSize,
        sequenceLength
    );

    if (gradAttention.device() == Device::CUDA) {
        cachedQ_.toCUDA();
        cachedK_.toCUDA();
        cachedV_.toCUDA();
        cachedAttentionWeights_.toCUDA();
        gradAttention.toCUDA();

        Tensor gradQ({ batchSize, sequenceLength, embedDim_ }, 0.0f);
        Tensor gradK({ batchSize, sequenceLength, embedDim_ }, 0.0f);
        Tensor gradV({ batchSize, sequenceLength, embedDim_ }, 0.0f);

        gradQ.toCUDA();
        gradK.toCUDA();
        gradV.toCUDA();

        launchSelfAttentionBackward(
            cachedQ_.deviceData(),
            cachedK_.deviceData(),
            cachedV_.deviceData(),
            cachedAttentionWeights_.deviceData(),
            gradAttention.deviceData(),
            gradQ.deviceData(),
            gradK.deviceData(),
            gradV.deviceData(),
            batchSize,
            sequenceLength,
            embedDim_
        );

        Tensor gradQFlat = LayerUtils::flatten3DTo2D(gradQ);
        Tensor gradKFlat = LayerUtils::flatten3DTo2D(gradK);
        Tensor gradVFlat = LayerUtils::flatten3DTo2D(gradV);

        Tensor gradInputQ = queryProj_.backward(gradQFlat);
        Tensor gradInputK = keyProj_.backward(gradKFlat);
        Tensor gradInputV = valueProj_.backward(gradVFlat);

        Tensor gradInputFlat = MathUtils::add(
            MathUtils::add(gradInputQ, gradInputK),
            gradInputV
        );

        return LayerUtils::unflatten2DTo3D(
            gradInputFlat,
            batchSize,
            sequenceLength
        );
    }

    Tensor gradQ({ batchSize, sequenceLength, embedDim_ }, 0.0f);
    Tensor gradK({ batchSize, sequenceLength, embedDim_ }, 0.0f);
    Tensor gradV({ batchSize, sequenceLength, embedDim_ }, 0.0f);

    float scale = 1.0f / std::sqrt(static_cast<float>(embedDim_));

    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t t = 0; t < sequenceLength; ++t) {
            std::vector<float> gradWeights(sequenceLength, 0.0f);

            for (size_t j = 0; j <= t; ++j) {
                float dot = 0.0f;

                for (size_t f = 0; f < embedDim_; ++f) {
                    dot += gradAttention.at({ b, t, f }) *
                        cachedV_.at({ b, j, f });

                    gradV.at({ b, j, f }) +=
                        cachedAttentionWeights_.at({ b, t, j }) *
                        gradAttention.at({ b, t, f });
                }

                gradWeights[j] = dot;
            }

            float weightedSum = 0.0f;

            for (size_t j = 0; j <= t; ++j) {
                weightedSum +=
                    gradWeights[j] *
                    cachedAttentionWeights_.at({ b, t, j });
            }

            for (size_t j = 0; j <= t; ++j) {
                float gradScore =
                    cachedAttentionWeights_.at({ b, t, j }) *
                    (gradWeights[j] - weightedSum);

                gradScore *= scale;

                for (size_t f = 0; f < embedDim_; ++f) {
                    gradQ.at({ b, t, f }) +=
                        gradScore * cachedK_.at({ b, j, f });

                    gradK.at({ b, j, f }) +=
                        gradScore * cachedQ_.at({ b, t, f });
                }
            }
        }
    }

    Tensor gradQFlat = LayerUtils::flatten3DTo2D(gradQ);
    Tensor gradKFlat = LayerUtils::flatten3DTo2D(gradK);
    Tensor gradVFlat = LayerUtils::flatten3DTo2D(gradV);

    Tensor gradInputQ = queryProj_.backward(gradQFlat);
    Tensor gradInputK = keyProj_.backward(gradKFlat);
    Tensor gradInputV = valueProj_.backward(gradVFlat);

    Tensor gradInputFlat = MathUtils::add(
        MathUtils::add(gradInputQ, gradInputK),
        gradInputV
    );

    return LayerUtils::unflatten2DTo3D(
        gradInputFlat,
        batchSize,
        sequenceLength
    );
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

Tensor MultiHeadAttention::forward(const Tensor& input){
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

    cachedInput_ = input;
    cachedQ_ = Q;
    cachedK_ = K;
    cachedV_ = V;
    cachedAttentionWeights_ = Tensor({ batchSize, numHeads_, sequenceLength, sequenceLength }, 0.0f);

    if (input.device() == Device::CUDA) {
        Q.toCUDA();
        K.toCUDA();
        V.toCUDA();
        cachedAttentionWeights_.toCUDA();

        Tensor attended({ batchSize, sequenceLength, embedDim_ }, 0.0f);
        attended.toCUDA();

        launchMultiHeadAttentionForward(
            cachedQ_.deviceData(),
            cachedK_.deviceData(),
            cachedV_.deviceData(),
            cachedAttentionWeights_.deviceData(),
            attended.deviceData(),
            batchSize,
            sequenceLength,
            numHeads_,
            headDim_
        );

        Tensor flatAttended = LayerUtils::flatten3DTo2D(attended);
        Tensor projectedFlat = outputProj_.forward(flatAttended);

        return LayerUtils::unflatten2DTo3D(
            projectedFlat,
            batchSize,
            sequenceLength
        );
    }

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

                for (size_t j = 0; j < sequenceLength; ++j) {
                    cachedAttentionWeights_.at({ b, h, t, j }) = weights[j];
                }

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

Tensor MultiHeadAttention::backward(const Tensor& gradOutput) {
    if (cachedInput_.empty()) {
        throw std::runtime_error("MultiHeadAttention::backward called before forward.");
    }

    if (gradOutput.rank() != 3) {
        throw std::invalid_argument("MultiHeadAttention::backward expects [batch, sequence, embedDim].");
    }

    size_t batchSize = gradOutput.shape()[0];
    size_t sequenceLength = gradOutput.shape()[1];

    Tensor flatGradOutput = LayerUtils::flatten3DTo2D(gradOutput);
    Tensor gradConcatFlat = outputProj_.backward(flatGradOutput);

    Tensor gradConcat = LayerUtils::unflatten2DTo3D(
        gradConcatFlat,
        batchSize,
        sequenceLength
    );

    if (gradConcat.device() == Device::CUDA) {
        cachedQ_.toCUDA();
        cachedK_.toCUDA();
        cachedV_.toCUDA();
        cachedAttentionWeights_.toCUDA();
        gradConcat.toCUDA();

        Tensor gradQ({ batchSize, sequenceLength, embedDim_ }, 0.0f);
        Tensor gradK({ batchSize, sequenceLength, embedDim_ }, 0.0f);
        Tensor gradV({ batchSize, sequenceLength, embedDim_ }, 0.0f);

        gradQ.toCUDA();
        gradK.toCUDA();
        gradV.toCUDA();

        launchMultiHeadAttentionBackward(
            cachedQ_.deviceData(),
            cachedK_.deviceData(),
            cachedV_.deviceData(),
            cachedAttentionWeights_.deviceData(),
            gradConcat.deviceData(),
            gradQ.deviceData(),
            gradK.deviceData(),
            gradV.deviceData(),
            batchSize,
            sequenceLength,
            numHeads_,
            headDim_
        );

        Tensor gradQFlat = LayerUtils::flatten3DTo2D(gradQ);
        Tensor gradKFlat = LayerUtils::flatten3DTo2D(gradK);
        Tensor gradVFlat = LayerUtils::flatten3DTo2D(gradV);

        Tensor gradInputQ = queryProj_.backward(gradQFlat);
        Tensor gradInputK = keyProj_.backward(gradKFlat);
        Tensor gradInputV = valueProj_.backward(gradVFlat);

        Tensor gradInputFlat = MathUtils::add(
            MathUtils::add(gradInputQ, gradInputK),
            gradInputV
        );

        return LayerUtils::unflatten2DTo3D(
            gradInputFlat,
            batchSize,
            sequenceLength
        );
    }

    Tensor gradQ({ batchSize, sequenceLength, embedDim_ }, 0.0f);
    Tensor gradK({ batchSize, sequenceLength, embedDim_ }, 0.0f);
    Tensor gradV({ batchSize, sequenceLength, embedDim_ }, 0.0f);

    float scale = 1.0f / std::sqrt(static_cast<float>(headDim_));

    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t h = 0; h < numHeads_; ++h) {
            for (size_t t = 0; t < sequenceLength; ++t) {
                std::vector<float> gradWeights(sequenceLength, 0.0f);

                for (size_t j = 0; j <= t; ++j) {
                    float dot = 0.0f;

                    for (size_t f = 0; f < headDim_; ++f) {
                        size_t globalF = h * headDim_ + f;

                        dot += gradConcat.at({ b, t, globalF }) *
                            cachedV_.at({ b, j, globalF });

                        gradV.at({ b, j, globalF }) +=
                            cachedAttentionWeights_.at({ b, h, t, j }) *
                            gradConcat.at({ b, t, globalF });
                    }

                    gradWeights[j] = dot;
                }

                float weightedSum = 0.0f;

                for (size_t j = 0; j <= t; ++j) {
                    weightedSum +=
                        gradWeights[j] *
                        cachedAttentionWeights_.at({ b, h, t, j });
                }

                for (size_t j = 0; j <= t; ++j) {
                    float gradScore =
                        cachedAttentionWeights_.at({ b, h, t, j }) *
                        (gradWeights[j] - weightedSum);

                    gradScore *= scale;

                    for (size_t f = 0; f < headDim_; ++f) {
                        size_t globalF = h * headDim_ + f;

                        gradQ.at({ b, t, globalF }) +=
                            gradScore * cachedK_.at({ b, j, globalF });

                        gradK.at({ b, j, globalF }) +=
                            gradScore * cachedQ_.at({ b, t, globalF });
                    }
                }
            }
        }
    }

    Tensor gradQFlat = LayerUtils::flatten3DTo2D(gradQ);
    Tensor gradKFlat = LayerUtils::flatten3DTo2D(gradK);
    Tensor gradVFlat = LayerUtils::flatten3DTo2D(gradV);

    Tensor gradInputQ = queryProj_.backward(gradQFlat);
    Tensor gradInputK = keyProj_.backward(gradKFlat);
    Tensor gradInputV = valueProj_.backward(gradVFlat);

    Tensor gradInputFlat = MathUtils::add(
        MathUtils::add(gradInputQ, gradInputK),
        gradInputV
    );

    return LayerUtils::unflatten2DTo3D(
        gradInputFlat,
        batchSize,
        sequenceLength
    );
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
