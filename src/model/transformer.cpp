#include "model/transformer.h"
#include "core/math_utils.h"
#include "core/layer_utils.h"
#include "core/parameter.h"

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
    outputHead_(config.block.d_model, config.vocab_size, rng) {

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

Tensor Transformer::forward(const Tensor& tokenIds) {
    if (tokenIds.rank() != 2) {
        throw std::invalid_argument("Transformer::forward expects tokenIds shape [batch, sequence].");
    }

    size_t batchSize = tokenIds.shape()[0];
    size_t sequenceLength = tokenIds.shape()[1];

    if (sequenceLength > maxSequenceLength_) {
        throw std::invalid_argument("Input sequence length exceeds maxSequenceLength.");
    }

    Tensor tokenEmbedded = tokenEmbedding_.forward(tokenIds);

    Tensor positionIds({ batchSize, sequenceLength }, 0.0f);

    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t t = 0; t < sequenceLength; ++t) {
            positionIds.at({ b, t }) = static_cast<float>(t);
        }
    }

    Tensor positionEmbedded = positionEmbedding_.forward(positionIds);

    Tensor x = MathUtils::add(tokenEmbedded, positionEmbedded);

    for (const TransformerBlock& block : blocks_) {
        x = block.forward(x);
    }

    x = finalNorm_.forward(x);

    Tensor flat = LayerUtils::flatten3DTo2D(x);
    Tensor logitsFlat = outputHead_.forward(flat);

    return LayerUtils::unflatten2DTo3D(logitsFlat, batchSize, sequenceLength);
}

std::string Transformer::generate(
    const std::string& prompt,
    const CharTokenizer& tokenizer,
    size_t maxNewTokens,
    float temperature,
    size_t topK,
    Random& rng
) {
    if (temperature <= 0.0f) {
        throw std::invalid_argument("Temperature must be > 0.");
    }

    if (topK == 0 || topK > vocabSize_) {
        topK = vocabSize_;
    }

    std::vector<size_t> tokenIds = tokenizer.encode(prompt);

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

    return tokenizer.decode(tokenIds);
}

std::string Transformer::generate(
    const std::string& prompt,
    const CharTokenizer& tokenizer,
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