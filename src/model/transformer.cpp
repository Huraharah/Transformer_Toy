#include "model/transformer.h"
#include "core/math_utils.h"

#include <stdexcept>
#include <algorithm>
#include <utility>

Transformer::Transformer(
    size_t vocabSize,
    size_t maxSequenceLength,
    size_t embedDim,
    size_t hiddenDim,
    size_t numLayers,
    Random& rng
)
    : vocabSize_(vocabSize),
    maxSequenceLength_(maxSequenceLength),
    embedDim_(embedDim),
    hiddenDim_(hiddenDim),
    numLayers_(numLayers),
    tokenEmbedding_(vocabSize, embedDim, rng),
    positionEmbedding_(maxSequenceLength, embedDim, rng),
    finalNorm_(embedDim),
    outputHead_(embedDim, vocabSize, rng) {

    blocks_.reserve(numLayers_);

    for (size_t i = 0; i < numLayers_; ++i) {
        blocks_.emplace_back(embedDim_, hiddenDim_, rng);
    }
}

Tensor Transformer::forward(const Tensor& tokenIds) const {
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
) const {
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