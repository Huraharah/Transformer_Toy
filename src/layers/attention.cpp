#include "layers/attention.h"
#include "core/math_utils.h"
#include "core/parameter.h"
#include "training/training_profiler.h"
#include "kernels/attention_kernels.cuh"

#include <cmath>
#include <vector>
#include <stdexcept>
#include <iostream>

SelfAttention::SelfAttention(const AttentionConfig& config, Random& rng)
    : config_(config),
    embedDim_(config.embedDim),
    queryProj_(config.embedDim, config.embedDim, rng, config.use_bias),
    keyProj_(config.embedDim, config.embedDim, rng, config.use_bias),
    valueProj_(config.embedDim, config.embedDim, rng, config.use_bias),
    outputProj_(config.embedDim, config.embedDim, rng, config.use_bias),
    attentionDropout_(config.attention_dropout, rng),
    projectionDropout_(config.projection_dropout, rng) {
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

    Tensor flatInput;
    Tensor qFlat;
    Tensor kFlat;
    Tensor vFlat;
    Tensor Q;
    Tensor K;
    Tensor V;
    Tensor attended;
    Tensor attentionOutput;
    Tensor projectedFlat;

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionFlattenForward, input.device());

        flatInput = input;
        flatInput.reshape({ batchSize * sequenceLength, embedDim_ });
    }

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionQProjectionForward, input.device());
        qFlat = queryProj_.forward(flatInput);
    }

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionKProjectionForward, input.device());
        kFlat = keyProj_.forward(flatInput);
    }

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionVProjectionForward, input.device());
        vFlat = valueProj_.forward(flatInput);
    }

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionCacheForward, input.device());

        qFlat.reshape({ batchSize, sequenceLength, embedDim_ });
        kFlat.reshape({ batchSize, sequenceLength, embedDim_ });
        vFlat.reshape({ batchSize, sequenceLength, embedDim_ });

        Q = std::move(qFlat);
        K = std::move(kFlat);
        V = std::move(vFlat);

        cachedInput_ = input;
        cachedQ_ = Q;
        cachedK_ = K;
        cachedV_ = V;
        cachedAttentionWeights_ = Tensor({ batchSize, sequenceLength, sequenceLength }, 0.0f);
    }

    if (input.device() == Device::CUDA) {
        {
            ScopedProfile profile(profiler_, ProfilePhase::AttentionKernelForward, input.device());

            cachedQ_.toCUDA();
            cachedK_.toCUDA();
            cachedV_.toCUDA();
            cachedAttentionWeights_.toCUDA();

            attended = Tensor({ batchSize, sequenceLength, embedDim_ }, 0.0f);
            attended.toCUDA();

            launchSelfAttentionWeightsForward(
                cachedQ_.deviceData(),
                cachedK_.deviceData(),
                cachedAttentionWeights_.deviceData(),
                batchSize,
                sequenceLength,
                embedDim_
            );

            cachedDroppedAttentionWeights_ =
                attentionDropout_.forward(cachedAttentionWeights_);

            launchSelfAttentionValuesForward(
                cachedV_.deviceData(),
                cachedDroppedAttentionWeights_.deviceData(),
                attended.deviceData(),
                batchSize,
                sequenceLength,
                embedDim_
            );

            {
                ScopedProfile profile(profiler_, ProfilePhase::AttentionOutputProjectionForward, attended.device());

                attended.reshape({ batchSize * sequenceLength, embedDim_ });
                projectedFlat = outputProj_.forward(attended);
                projectedFlat = projectionDropout_.forward(projectedFlat);
            }

            {
                ScopedProfile profile(profiler_, ProfilePhase::AttentionUnflattenForward, projectedFlat.device());

                projectedFlat.reshape({ batchSize, sequenceLength, embedDim_ });
            }

            return projectedFlat;
        }
    }

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionKernelForward, input.device());
        attentionOutput = Tensor({ batchSize, sequenceLength, embedDim_ }, 0.0f);

        float scale = 1.0f / std::sqrt(static_cast<float>(embedDim_));

        for (size_t b = 0; b < batchSize; ++b) {
            for (size_t t = 0; t < sequenceLength; ++t) {
                std::vector<float> scores(sequenceLength, -1.0e9f);

                size_t keyLimit = config_.causal ? t + 1 : sequenceLength;

                for (size_t j = 0; j < keyLimit; ++j) {
                    float dot = 0.0f;

                    for (size_t f = 0; f < embedDim_; ++f) {
                        size_t idx = f;
                        dot += Q.at({ b, t, idx }) * K.at({ b, j, idx });
                    }

                    scores[j] = dot * scale;
                }

                std::vector<float> weights = MathUtils::softmax(scores);

                for (size_t j = 0; j < sequenceLength; ++j) {
                    cachedAttentionWeights_.at({ b,  t, j }) = weights[j];
                }
            }
            
        }

        cachedDroppedAttentionWeights_ = attentionDropout_.forward(cachedAttentionWeights_);

        for (size_t b = 0; b < batchSize; ++b) {
            for (size_t t = 0; t < sequenceLength; ++t) {
                for (size_t f = 0; f < embedDim_; ++f) {
                    size_t idx = f;
                    float sum = 0.0f;

                    size_t keyLimit = config_.causal ? t + 1 : sequenceLength;

                    for (size_t j = 0; j < keyLimit; ++j) {
                        sum += cachedDroppedAttentionWeights_.at({ b, t, j }) * V.at({ b, j, idx });
                    }

                    attentionOutput.at({ b, t, idx }) = sum;
                }
            }
            
        }

        {
            ScopedProfile profile(profiler_, ProfilePhase::AttentionOutputProjectionForward, attentionOutput.device());

            attentionOutput.reshape({ batchSize * sequenceLength, embedDim_ });
            projectedFlat = outputProj_.forward(attentionOutput);
            projectedFlat = projectionDropout_.forward(projectedFlat);
        }

        {
            ScopedProfile profile(profiler_, ProfilePhase::AttentionUnflattenForward, projectedFlat.device());

            projectedFlat.reshape({ batchSize, sequenceLength, embedDim_ });
        }

        return projectedFlat;
    }
}

Tensor SelfAttention::backward(const Tensor& gradOutput) {
    if (cachedInput_.empty()) {
        throw std::runtime_error("SelfAttention::backward called before forward.");
    }

    size_t batchSize = gradOutput.shape()[0];
    size_t sequenceLength = gradOutput.shape()[1];

    Tensor flatGradOutput;
    Tensor gradConcatFlat;
    Tensor gradConcat;

    Tensor gradQ;
    Tensor gradK;
    Tensor gradV;

    Tensor gradInputQ;
    Tensor gradInputK;
    Tensor gradInputV;

    Tensor gradInputFlat;

    Tensor gradDroppedWeights({ batchSize, sequenceLength, sequenceLength }, 0.0f);
    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionFlattenBackward, gradOutput.device());

        flatGradOutput = gradOutput;
        flatGradOutput.reshape({ batchSize * sequenceLength, embedDim_ });
    }

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionOutputProjectionBackward, gradOutput.device());

        flatGradOutput = projectionDropout_.backward(flatGradOutput);
        gradConcatFlat = outputProj_.backward(flatGradOutput);
        gradConcatFlat.reshape({ batchSize, sequenceLength, embedDim_ });
        gradConcat = std::move(gradConcatFlat);
    }

    if (gradConcat.device() == Device::CUDA) {

        {
            ScopedProfile profile(profiler_, ProfilePhase::AttentionKernelBackward, gradConcat.device());

            cachedQ_.toCUDA();
            cachedK_.toCUDA();
            cachedV_.toCUDA();
            cachedAttentionWeights_.toCUDA();
            gradConcat.toCUDA();

            gradQ = Tensor({ batchSize, sequenceLength, embedDim_ }, 0.0f);
            gradK = Tensor({ batchSize, sequenceLength, embedDim_ }, 0.0f);
            gradV = Tensor({ batchSize, sequenceLength, embedDim_ }, 0.0f);

            gradQ.toCUDA();
            gradK.toCUDA();
            gradV.toCUDA();

            Tensor gradDroppedWeights(
                { batchSize, sequenceLength, sequenceLength },
                0.0f
            );

            gradDroppedWeights.toCUDA();

            launchSelfAttentionBackwardValues(
                cachedV_.deviceData(),
                cachedDroppedAttentionWeights_.deviceData(),
                gradConcat.deviceData(),
                gradDroppedWeights.deviceData(),
                gradV.deviceData(),
                batchSize,
                sequenceLength,
                embedDim_
            );

            Tensor gradAttentionWeights =
                attentionDropout_.backward(gradDroppedWeights);

            launchSelfAttentionBackwardWeights(
                cachedQ_.deviceData(),
                cachedK_.deviceData(),
                cachedAttentionWeights_.deviceData(),
                gradAttentionWeights.deviceData(),
                gradQ.deviceData(),
                gradK.deviceData(),
                batchSize,
                sequenceLength,
                embedDim_
            );

            gradQ.reshape({ batchSize * sequenceLength, embedDim_ });
            gradK.reshape({ batchSize * sequenceLength, embedDim_ });
            gradV.reshape({ batchSize * sequenceLength, embedDim_ });

            {
                ScopedProfile profile(profiler_, ProfilePhase::AttentionQProjectionBackward, gradQ.device());
                gradInputQ = queryProj_.backward(gradQ);
            }

            {
                ScopedProfile profile(profiler_, ProfilePhase::AttentionKProjectionBackward, gradK.device());
                gradInputK = keyProj_.backward(gradK);
            }

            {
                ScopedProfile profile(profiler_, ProfilePhase::AttentionVProjectionBackward, gradV.device());
                gradInputV = valueProj_.backward(gradV);
            }

            gradInputFlat = MathUtils::add(
                MathUtils::add(gradInputQ, gradInputK),
                gradInputV
            );

            gradInputFlat.reshape({ batchSize, sequenceLength, embedDim_ });
            return gradInputFlat;
        }
    }

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionKernelBackward, gradConcat.device());

        gradQ = Tensor({ batchSize, sequenceLength, embedDim_ }, 0.0f);
        gradK = Tensor({ batchSize, sequenceLength, embedDim_ }, 0.0f);
        gradV = Tensor({ batchSize, sequenceLength, embedDim_ }, 0.0f);

        float scale = 1.0f / std::sqrt(static_cast<float>(embedDim_));

        for (size_t b = 0; b < batchSize; ++b) {
            for (size_t t = 0; t < sequenceLength; ++t) {
                size_t keyLimit = config_.causal ? t + 1 : sequenceLength;

                for (size_t j = 0; j < keyLimit; ++j) {
                    float dot = 0.0f;

                    for (size_t f = 0; f < embedDim_; ++f) {
                        size_t idx = f;

                        dot +=
                            gradConcat.at({ b, t, idx }) *
                            cachedV_.at({ b, j, idx });

                        gradV.at({ b, j, idx }) +=
                            cachedDroppedAttentionWeights_.at({ b, t, j }) *
                            gradConcat.at({ b, t, idx });
                    }

                    gradDroppedWeights.at({ b, t, j }) = dot;
                }
            }
        }

        Tensor gradAttentionWeights = attentionDropout_.backward(gradDroppedWeights);

        for (size_t b = 0; b < batchSize; ++b) {
            for (size_t t = 0; t < sequenceLength; ++t) {
                float weightedSum = 0.0f;
                size_t keyLimit = config_.causal ? t + 1 : sequenceLength;

                for (size_t j = 0; j < keyLimit; ++j) {
                    weightedSum +=
                        gradAttentionWeights.at({ b, t, j }) *
                        cachedAttentionWeights_.at({ b, t, j });
                }

                for (size_t j = 0; j < keyLimit; ++j) {
                    float gradScore =
                        cachedAttentionWeights_.at({ b, t, j }) *
                        (
                            gradAttentionWeights.at({ b, t, j }) -
                            weightedSum
                            );

                    gradScore *= scale;

                    for (size_t f = 0; f < embedDim_; ++f) {
                        size_t idx = f;

                        gradQ.at({ b, t, idx }) +=
                            gradScore * cachedK_.at({ b, j, idx });

                        gradK.at({ b, j, idx }) +=
                            gradScore * cachedQ_.at({ b, t, idx });
                    }
                }
            }
        }

        gradQ.reshape({ batchSize * sequenceLength, embedDim_ });
        gradK.reshape({ batchSize * sequenceLength, embedDim_ });
        gradV.reshape({ batchSize * sequenceLength, embedDim_ });

        {
            ScopedProfile profile(profiler_, ProfilePhase::AttentionQProjectionBackward, gradQ.device());
            gradInputQ = queryProj_.backward(gradQ);
        }

        {
            ScopedProfile profile(profiler_, ProfilePhase::AttentionKProjectionBackward, gradK.device());
            gradInputK = keyProj_.backward(gradK);
        }

        {
            ScopedProfile profile(profiler_, ProfilePhase::AttentionVProjectionBackward, gradV.device());
            gradInputV = valueProj_.backward(gradV);
        }

        {
            ScopedProfile profile(profiler_, ProfilePhase::AttentionGradientMergeBackward, gradInputQ.device());

            gradInputFlat = MathUtils::add(
                MathUtils::add(gradInputQ, gradInputK),
                gradInputV
            );
        }

        {
            ScopedProfile profile(profiler_, ProfilePhase::AttentionUnflattenBackward, gradInputFlat.device());

            gradInputFlat.reshape({ batchSize, sequenceLength, embedDim_ });
        }
        return gradInputFlat;
    }
}

void SelfAttention::setProfiler(TrainingProfiler* profiler) {
    profiler_ = profiler;
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

void SelfAttention::train() {
    training_ = true;
    attentionDropout_.train();
    projectionDropout_.train();
}

void SelfAttention::eval() {
    training_ = false;
    attentionDropout_.eval();
    projectionDropout_.eval();
}

bool SelfAttention::isTraining() const {
    return training_;
}

MultiHeadAttention::MultiHeadAttention(const AttentionConfig& config, Random& rng)
    : config_(config),
    embedDim_(config.embedDim),
    numHeads_(config.numHeads),
    headDim_(config.embedDim / config.numHeads),
    queryProj_(config.embedDim, config.embedDim, rng, config.use_bias),
    keyProj_(config.embedDim, config.embedDim, rng, config.use_bias),
    valueProj_(config.embedDim, config.embedDim, rng, config.use_bias),
    outputProj_(config.embedDim, config.embedDim, rng, config.use_bias),
    attentionDropout_(config.attention_dropout, rng),
    projectionDropout_(config.projection_dropout, rng) {
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

    Tensor flatInput;
    Tensor qFlat;
    Tensor kFlat;
    Tensor vFlat;
    Tensor Q;
    Tensor K;
    Tensor V;
    Tensor attended;
    Tensor attentionOutput;
    Tensor projectedFlat;

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionFlattenForward, input.device());

        flatInput = input;
        flatInput.reshape({ batchSize * sequenceLength, embedDim_ });
    }

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionQProjectionForward, input.device());
        qFlat = queryProj_.forward(flatInput);
    }

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionKProjectionForward, input.device());
        kFlat = keyProj_.forward(flatInput);
    }

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionVProjectionForward, input.device());
        vFlat = valueProj_.forward(flatInput);
    }

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionCacheForward, input.device());

        qFlat.reshape({ batchSize, sequenceLength, embedDim_ });
        kFlat.reshape({ batchSize, sequenceLength, embedDim_ });
        vFlat.reshape({ batchSize, sequenceLength, embedDim_ });

        Q = std::move(qFlat);
        K = std::move(kFlat);
        V = std::move(vFlat);

        cachedInput_ = input;
        cachedQ_ = Q;
        cachedK_ = K;
        cachedV_ = V;
        cachedAttentionWeights_ = Tensor({ batchSize, numHeads_, sequenceLength, sequenceLength }, 0.0f);
    }

    if (input.device() == Device::CUDA) {
        cachedQ_.toCUDA();
        cachedK_.toCUDA();
        cachedV_.toCUDA();
        cachedAttentionWeights_.toCUDA();

        attended = Tensor({ batchSize, sequenceLength, embedDim_ }, 0.0f);
        attended.toCUDA();

        {
            ScopedProfile profile(profiler_, ProfilePhase::AttentionKernelForward, input.device());

            launchMultiHeadAttentionWeightsForward(
                cachedQ_.deviceData(),
                cachedK_.deviceData(),
                cachedAttentionWeights_.deviceData(),
                batchSize,
                sequenceLength,
                numHeads_,
                headDim_
            );

            cachedDroppedAttentionWeights_ = attentionDropout_.forward(cachedAttentionWeights_);

            launchMultiHeadAttentionValuesForward(
                cachedV_.deviceData(),
                cachedDroppedAttentionWeights_.deviceData(),
                attended.deviceData(),
                batchSize,
                sequenceLength,
                numHeads_,
                headDim_
            );

        }

        {
            ScopedProfile profile(profiler_, ProfilePhase::AttentionOutputProjectionForward, attended.device());

            attended.reshape({ batchSize * sequenceLength, embedDim_ });
            projectedFlat = outputProj_.forward(attended);
            projectedFlat = projectionDropout_.forward(projectedFlat);
        }

        {
            ScopedProfile profile(profiler_, ProfilePhase::AttentionUnflattenForward, projectedFlat.device());

            projectedFlat.reshape({ batchSize, sequenceLength, embedDim_ });
        }

        return projectedFlat;
        
    }

    attentionOutput = Tensor({ batchSize, sequenceLength, embedDim_ }, 0.0f);

    float scale = 1.0f / std::sqrt(static_cast<float>(headDim_));

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionKernelForward, input.device());

        for (size_t b = 0; b < batchSize; ++b) {
            for (size_t h = 0; h < numHeads_; ++h) {
                for (size_t t = 0; t < sequenceLength; ++t) {
                    std::vector<float> scores(sequenceLength, -1.0e9f);
                    size_t keyLimit = config_.causal ? t + 1 : sequenceLength;

                    for (size_t j = 0; j < keyLimit; ++j) {
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
                }
            }
        }

        cachedDroppedAttentionWeights_ = attentionDropout_.forward(cachedAttentionWeights_);

        for (size_t b = 0; b < batchSize; ++b) {
            for (size_t h = 0; h < numHeads_; ++h) {
                for (size_t t = 0; t < sequenceLength; ++t) {
                    for (size_t f = 0; f < headDim_; ++f) {
                        size_t idx = h * headDim_ + f;
                        float sum = 0.0f;
                        size_t keyLimit = config_.causal ? t + 1 : sequenceLength;

                        for (size_t j = 0; j < keyLimit; ++j) {
                            sum += cachedDroppedAttentionWeights_.at({ b, h, t, j }) * V.at({ b, j, idx });
                        }

                        attentionOutput.at({ b, t, idx }) = sum;
                    }
                }
            }
        }
    }

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionOutputProjectionForward, attentionOutput.device());

        attentionOutput.reshape({ batchSize * sequenceLength, embedDim_ });
        projectedFlat = outputProj_.forward(attentionOutput);
        projectedFlat = projectionDropout_.forward(projectedFlat);
    }

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionUnflattenForward, projectedFlat.device());

        projectedFlat.reshape({ batchSize, sequenceLength, embedDim_ });
    }

        return projectedFlat;
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

    Tensor flatGradOutput;
    Tensor gradConcatFlat;
    Tensor gradConcat;

    Tensor gradQ;
    Tensor gradK;
    Tensor gradV;

    Tensor gradInputQ;
    Tensor gradInputK;
    Tensor gradInputV;

    Tensor gradInputFlat;

    Tensor gradDroppedWeights({ batchSize, numHeads_, sequenceLength, sequenceLength }, 0.0f);

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionFlattenBackward, gradOutput.device());

        flatGradOutput = gradOutput;
        flatGradOutput.reshape({ batchSize * sequenceLength, embedDim_ });
    }

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionOutputProjectionBackward, gradOutput.device());

        flatGradOutput = projectionDropout_.backward(flatGradOutput);
        gradConcatFlat = outputProj_.backward(flatGradOutput);
        gradConcatFlat.reshape({ batchSize, sequenceLength, embedDim_ });
        gradConcat = std::move(gradConcatFlat);
    }

    if (gradConcat.device() == Device::CUDA) {
        cachedQ_.toCUDA();
        cachedK_.toCUDA();
        cachedV_.toCUDA();
        cachedAttentionWeights_.toCUDA();
        gradConcat.toCUDA();

        gradQ = Tensor({ batchSize, sequenceLength, embedDim_ }, 0.0f);
        gradK = Tensor({ batchSize, sequenceLength, embedDim_ }, 0.0f);
        gradV = Tensor({ batchSize, sequenceLength, embedDim_ }, 0.0f);

        gradQ.toCUDA();
        gradK.toCUDA();
        gradV.toCUDA();

        {
            ScopedProfile profile(profiler_, ProfilePhase::AttentionKernelBackward, gradConcat.device());

            cachedDroppedAttentionWeights_.toCUDA();
            gradDroppedWeights.toCUDA();

            launchMultiHeadAttentionBackwardValues(
                cachedV_.deviceData(),
                cachedDroppedAttentionWeights_.deviceData(),
                gradConcat.deviceData(),
                gradDroppedWeights.deviceData(),
                gradV.deviceData(),
                batchSize,
                sequenceLength,
                numHeads_,
                headDim_
            );

            Tensor gradAttentionWeights = attentionDropout_.backward(gradDroppedWeights);
        
            launchMultiHeadAttentionBackwardWeights(
                cachedQ_.deviceData(),
                cachedK_.deviceData(),
                cachedAttentionWeights_.deviceData(),
                gradConcat.deviceData(),
                gradQ.deviceData(),
                gradK.deviceData(),
                batchSize,
                sequenceLength,
                numHeads_,
                headDim_
            );
        }

        gradQ.reshape({ batchSize * sequenceLength, embedDim_ });
        gradK.reshape({ batchSize * sequenceLength, embedDim_ });
        gradV.reshape({ batchSize * sequenceLength, embedDim_ });

        {
            ScopedProfile profile(profiler_, ProfilePhase::AttentionQProjectionBackward, gradQ.device());
            gradInputQ = queryProj_.backward(gradQ);
        }

        {
            ScopedProfile profile(profiler_, ProfilePhase::AttentionKProjectionBackward, gradK.device());
            gradInputK = keyProj_.backward(gradK);
        }

        {
            ScopedProfile profile(profiler_, ProfilePhase::AttentionVProjectionBackward, gradV.device());
            gradInputV = valueProj_.backward(gradV);
        }

        {
            ScopedProfile profile(profiler_, ProfilePhase::AttentionGradientMergeBackward, gradInputQ.device());

            gradInputFlat = MathUtils::add(
                MathUtils::add(gradInputQ, gradInputK),
                gradInputV
            );
        }

        {
            ScopedProfile profile(profiler_, ProfilePhase::AttentionUnflattenBackward, gradInputFlat.device());

            gradInputFlat.reshape({ batchSize, sequenceLength, embedDim_ });
        }
        return gradInputFlat;
    }

    gradQ = Tensor({ batchSize, sequenceLength, embedDim_ }, 0.0f);
    gradK = Tensor({ batchSize, sequenceLength, embedDim_ }, 0.0f);
    gradV = Tensor({ batchSize, sequenceLength, embedDim_ }, 0.0f);

    float scale = 1.0f / std::sqrt(static_cast<float>(headDim_));

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionKernelBackward, gradConcat.device());

        for (size_t b = 0; b < batchSize; ++b) {
            for (size_t h = 0; h < numHeads_; ++h) {
                for (size_t t = 0; t < sequenceLength; ++t) {
                    size_t keyLimit = config_.causal ? t + 1 : sequenceLength;

                    for (size_t j = 0; j < keyLimit; ++j) {
                        float dot = 0.0f;

                        for (size_t f = 0; f < headDim_; ++f) {
                            size_t idx = h * headDim_ + f;

                            dot +=
                                gradConcat.at({ b, t, idx }) *
                                cachedV_.at({ b, j, idx });

                            gradV.at({ b, j, idx }) +=
                                cachedDroppedAttentionWeights_.at({ b, h, t, j }) *
                                gradConcat.at({ b, t, idx });
                        }

                        gradDroppedWeights.at({ b, h, t, j }) = dot;
                    }
                }
            }
        }

        Tensor gradAttentionWeights = attentionDropout_.backward(gradDroppedWeights);

        for (size_t b = 0; b < batchSize; ++b) {
            for (size_t h = 0; h < numHeads_; ++h) {
                for (size_t t = 0; t < sequenceLength; ++t) {
                    float weightedSum = 0.0f;
                    size_t keyLimit = config_.causal ? t + 1 : sequenceLength;

                    for (size_t j = 0; j < keyLimit; ++j) {
                        weightedSum +=
                            gradAttentionWeights.at({ b, h, t, j }) *
                            cachedAttentionWeights_.at({ b, h, t, j });
                    }

                    for (size_t j = 0; j < keyLimit; ++j) {
                        float gradScore =
                            cachedAttentionWeights_.at({ b, h, t, j }) *
                            (
                                gradAttentionWeights.at({ b, h, t, j }) -
                                weightedSum
                                );

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
    }

    gradQ.reshape({ batchSize * sequenceLength, embedDim_ });
    gradK.reshape({ batchSize * sequenceLength, embedDim_ });
    gradV.reshape({ batchSize * sequenceLength, embedDim_ });

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionQProjectionBackward, gradQ.device());
        gradInputQ = queryProj_.backward(gradQ);
    }

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionKProjectionBackward, gradK.device());
        gradInputK = keyProj_.backward(gradK);
    }

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionVProjectionBackward, gradV.device());
        gradInputV = valueProj_.backward(gradV);
    }

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionGradientMergeBackward, gradInputQ.device());

        gradInputFlat = MathUtils::add(
            MathUtils::add(gradInputQ, gradInputK),
            gradInputV
        );
    }

    {
        ScopedProfile profile(profiler_, ProfilePhase::AttentionUnflattenBackward, gradInputFlat.device());

        gradInputFlat.reshape({ batchSize, sequenceLength, embedDim_ });
    }
        return gradInputFlat;
}

void MultiHeadAttention::setProfiler(TrainingProfiler* profiler) {
    profiler_ = profiler;
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

void MultiHeadAttention::train() {
    training_ = true;
    attentionDropout_.train();
    projectionDropout_.train();
}

void MultiHeadAttention::eval() {
    training_ = false;
    attentionDropout_.eval();
    projectionDropout_.eval();
}

bool MultiHeadAttention::isTraining() const {
    return training_;
}