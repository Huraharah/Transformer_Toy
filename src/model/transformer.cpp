#include "model/transformer.h"
#include "core/math_utils.h"
#include "core/layer_utils.h"
#include "core/parameter.h"
#include "layers/dropout.h"

#include <stdexcept>
#include <algorithm>
#include <utility>

Transformer::Transformer(const TransformerModelConfig& config, Random& rng)
    : config_(config),
	maxSequenceLength_(config.max_seq_len),
    vocabSize_(config.vocab_size),
    contextLength_(config.max_seq_len),
    embedDim_(config.block.d_model),
    hiddenDim_(config.block.d_ff),
    numLayers_(config.num_layers),
    tokenEmbedding_(config.vocab_size, config.block.d_model, rng),
    positionEmbedding_(config.max_seq_len, config.block.d_model, rng),
    finalNorm_(config.block.d_model),
    outputHead_(config.block.d_model, config.vocab_size, rng),
    embeddingDropout_(config.embedding_dropout, rng) {

    config_.validate();

    blocks_.reserve(numLayers_);

    for (size_t i = 0; i < numLayers_; ++i) {
        blocks_.emplace_back(config_.block, rng);
    }
}

Transformer::Transformer(
    size_t vocabSize,
    size_t contextLength,
    size_t embedDim,
    size_t hiddenDim,
    size_t numLayers,
    Random& rng
)
    : Transformer(
        TransformerModelConfig(
            static_cast<int>(vocabSize),
            static_cast<int>(contextLength),
            static_cast<int>(numLayers),
            TransformerBlockConfig(
                static_cast<int>(embedDim),
                static_cast<int>(hiddenDim)
            )
        ),
        rng
    ) {
}

void Transformer::setProfiler(TrainingProfiler* profiler) {
    profiler_ = profiler;

    for (TransformerBlock& block : blocks_) {
        block.setProfiler(profiler);
    }
}

Tensor Transformer::forward(
    const Tensor& tokenIds
) {
    if (tokenIds.rank() != 2) {
        throw std::invalid_argument(
            "Transformer::forward expects tokenIds shape "
            "[batch, sequence]."
        );
    }

    const size_t batchSize =
        tokenIds.shape()[0];

    const size_t sequenceLength =
        tokenIds.shape()[1];

    if (sequenceLength > maxSequenceLength_) {
        throw std::invalid_argument(
            "Input sequence length exceeds maxSequenceLength."
        );
    }

    Tensor tokenEmbedded;
    Tensor positionIds;
    Tensor positionEmbedded;
    Tensor x;
    Tensor flat;
    Tensor logitsFlat;

    // ========================================================
    // Token and positional embeddings
    // ========================================================

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::TransformerEmbedding,
            tokenIds.device()
        );

        tokenEmbedded =
            tokenEmbedding_.forward(tokenIds);

        positionIds = Tensor(
            { batchSize, sequenceLength },
            0.0f
        );

        for (
            size_t batch = 0;
            batch < batchSize;
            ++batch
            ) {
            for (
                size_t position = 0;
                position < sequenceLength;
                ++position
                ) {
                positionIds.at({
                    batch,
                    position
                    }) = static_cast<float>(
                        position
                        );
            }
        }

        if (tokenIds.device() == Device::CUDA) {
            positionIds.toCUDA();
        }

        positionEmbedded =
            positionEmbedding_.forward(
                positionIds
            );

        x = MathUtils::add(tokenEmbedded, positionEmbedded);
        x = embeddingDropout_.forward(x);
    }

    // ========================================================
    // Transformer blocks
    // ========================================================

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::TransformerBlocksForward,
            tokenIds.device()
        );

        for (TransformerBlock& block : blocks_) {
            x = block.forward(x);
        }
    }

    // ========================================================
    // Final normalization
    // ========================================================

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::TransformerFinalNorm,
            tokenIds.device()
        );

        x = finalNorm_.forward(x);
    }

    // ========================================================
    // Output projection
    // ========================================================

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::TransformerOutputProjection,
            tokenIds.device()
        );

        flat =
            LayerUtils::flatten3DTo2D(x);

        logitsFlat =
            outputHead_.forward(flat);
    }

    return LayerUtils::unflatten2DTo3D(
        logitsFlat,
        batchSize,
        sequenceLength
    );
}

void Transformer::backward(
    const Tensor& gradOutput
) {
    if (gradOutput.rank() != 3) {
        throw std::invalid_argument(
            "Transformer::backward expects gradOutput shape "
            "[batch, sequence, vocabSize]."
        );
    }

    const size_t batchSize =
        gradOutput.shape()[0];

    const size_t sequenceLength =
        gradOutput.shape()[1];

    const size_t gradVocabSize =
        gradOutput.shape()[2];

    if (gradVocabSize != vocabSize_) {
        throw std::invalid_argument(
            "Transformer::backward vocab size mismatch."
        );
    }

    Tensor flatGradOutput;
    Tensor gradHiddenFlat;
    Tensor gradHidden;
    Tensor grad;

    // ========================================================
    // Output projection backward
    // ========================================================

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::TransformerOutputBackward,
            gradOutput.device()
        );

        flatGradOutput =
            LayerUtils::flatten3DTo2D(
                gradOutput
            );

        gradHiddenFlat =
            outputHead_.backward(
                flatGradOutput
            );

        gradHidden =
            LayerUtils::unflatten2DTo3D(
                gradHiddenFlat,
                batchSize,
                sequenceLength
            );
    }

    // ========================================================
    // Final normalization backward
    // ========================================================

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::TransformerFinalNormBackward,
            gradOutput.device()
        );

        grad =
            finalNorm_.backward(
                gradHidden
            );
    }

    // ========================================================
    // Transformer blocks backward
    // ========================================================

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::TransformerBlocksBackward,
            gradOutput.device()
        );

        for (auto iterator = blocks_.rbegin(); iterator != blocks_.rend(); ++iterator) {
            grad = iterator->backward(grad);
        }

        grad = embeddingDropout_.backward(grad);
    }

    // ========================================================
    // Embedding backward
    // ========================================================

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::TransformerEmbeddingBackward,
            gradOutput.device()
        );

        tokenEmbedding_.backward(
            grad
        );

        positionEmbedding_.backward(
            grad
        );
    }
}

std::string Transformer::generate(
    const std::string& prompt,
    const Tokenizer& tokenizer,
    size_t maxNewTokens,
    float temperature,
    size_t topK,
    Random& rng
) {
    const bool wasTraining = isTraining();

    eval();

    try {
        if (temperature <= 0.0f) {
            throw std::invalid_argument("Temperature must be > 0.");
        }

        if (topK == 0 || topK > vocabSize_) {
            topK = vocabSize_;
        }

        std::vector<int> tokenIds = tokenizer.encode(prompt);

        for (size_t step = 0; step < maxNewTokens; ++step) {
            size_t start = 0;

            if (tokenIds.size() > maxSequenceLength_) {
                start = tokenIds.size() - maxSequenceLength_;
            }

            size_t currentLength = tokenIds.size() - start;

            Tensor input({ 1, currentLength }, 0.0f);

            for (size_t i = 0; i < currentLength; ++i) {
                input.at({ 0, i }) =
                    static_cast<float>(tokenIds[start + i]);
            }

            Tensor logits = forward(input);

            size_t lastPosition = currentLength - 1;

            std::vector<std::pair<float, size_t>> candidates;
            candidates.reserve(vocabSize_);

            for (size_t v = 0; v < vocabSize_; ++v) {
                float scaledLogit =
                    logits.at({ 0, lastPosition, v }) / temperature;

                candidates.push_back({ scaledLogit, v });
            }

            std::sort(
                candidates.begin(),
                candidates.end(),
                [](const auto& a, const auto& b) {
                    return a.first > b.first;
                }
            );

            std::vector<float> topLogits;
            std::vector<size_t> topIds;

            for (size_t i = 0; i < topK; ++i) {
                topLogits.push_back(candidates[i].first);
                topIds.push_back(candidates[i].second);
            }

            std::vector<float> probabilities =
                MathUtils::softmax(topLogits);

            float sample = rng.uniform(0.0f, 1.0f);

            float cumulative = 0.0f;
            size_t selectedId = topIds.back();

            for (size_t i = 0; i < probabilities.size(); ++i) {
                cumulative += probabilities[i];

                if (sample <= cumulative) {
                    selectedId = topIds[i];
                    break;
                }
            }

            tokenIds.push_back(selectedId);
        }

        std::string result = tokenizer.decode(tokenIds);

        if (wasTraining) {
            train();
        }
            return result;
    }
    catch (...) {
        if (wasTraining) {
            train();
        }

        throw;
    }
}

std::string Transformer::generate(
    const std::string& prompt,
    const Tokenizer& tokenizer,
    const GenerationConfig& config,
    Random& rng
) {
    config.validate();

    return generate(
        prompt,
        tokenizer,
        config.maxNewTokens,
        config.temperature,
        config.topK,
        rng
    );
}

std::vector<Parameter*> Transformer::parameters() {
    std::vector<Parameter*> params;

    auto append = [&params](std::vector<Parameter*> more) {
        params.insert(params.end(), more.begin(), more.end());
        };

    append(tokenEmbedding_.parameters());
    append(positionEmbedding_.parameters());

    for (TransformerBlock& block : blocks_) {
        append(block.parameters());
    }

    append(finalNorm_.parameters());
    append(outputHead_.parameters());

    return params;
}

void Transformer::train() {
    training_ = true;
    embeddingDropout_.train();

    for (TransformerBlock& block : blocks_) {
        block.train();
    }
}

void Transformer::eval() {
    training_ = false;
    embeddingDropout_.eval();

    for (TransformerBlock& block : blocks_) {
        block.eval();
    }
}

bool Transformer::isTraining() const {
    return training_;
}