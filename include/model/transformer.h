#pragma once

#include <vector>

#include "core/tensor.h"
#include "core/random.h"
#include "layers/embedding.h"
#include "layers/transformer_block.h"
#include "layers/layer_norm.h"
#include "layers/linear.h"
#include "data/tokenizer.h"
#include "core/layer_utils.h"

class Transformer {
private:
    size_t vocabSize_;
    size_t maxSequenceLength_;
    size_t embedDim_;
    size_t hiddenDim_;
    size_t numLayers_;

    Embedding tokenEmbedding_;
    Embedding positionEmbedding_;

    std::vector<TransformerBlock> blocks_;

    LayerNorm finalNorm_;
    Linear outputHead_;

public:
    Transformer(
        size_t vocabSize,
        size_t maxSequenceLength,
        size_t embedDim,
        size_t hiddenDim,
        size_t numLayers,
        Random& rng
    );

    std::string generate(
        const std::string& prompt,
        const CharTokenizer& tokenizer,
        size_t maxNewTokens,
        float temperature,
        size_t topK,
        Random& rng
    ) const;

    Tensor forward(const Tensor& tokenIds) const;
};