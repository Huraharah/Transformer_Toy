#include "layers/embedding.h"

#include <cmath>
#include <stdexcept>

Embedding::Embedding(size_t vocabSize, size_t embeddingDim, Random& rng)
    : vocabSize_(vocabSize),
    embeddingDim_(embeddingDim),
    table_(Tensor({ vocabSize, embeddingDim }), "embedding.table") {

    float limit = std::sqrt(6.0f / static_cast<float>(vocabSize + embeddingDim));

    for (size_t i = 0; i < table_.size(); ++i) {
        table_.value[i] = rng.uniform(-limit, limit);
    }
}

Tensor Embedding::forward(const Tensor& tokenIds) {
    if (tokenIds.rank() != 2) {
        throw std::invalid_argument("Embedding::forward expects input shape [batchSize, sequenceLength].");
    }

	cachedTokenIds_ = tokenIds;

    size_t batchSize = tokenIds.shape()[0];
    size_t sequenceLength = tokenIds.shape()[1];

    Tensor output({ batchSize, sequenceLength, embeddingDim_ }, 0.0f);

    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t t = 0; t < sequenceLength; ++t) {
            int tokenId = static_cast<int>(tokenIds.at({ b, t }));

            if (tokenId < 0 || static_cast<size_t>(tokenId) >= vocabSize_) {
                throw std::out_of_range("Embedding token ID out of range.");
            }

            for (size_t e = 0; e < embeddingDim_; ++e) {
                output.at({ b, t, e }) = table_.value.at({ static_cast<size_t>(tokenId), e });
            }
        }
    }

    return output;
}

Tensor Embedding::backward(const Tensor& gradOutput) {
    if (cachedTokenIds_.empty()) {
        throw std::runtime_error("Embedding::backward called before forward.");
    }

    if (gradOutput.rank() != 3) {
        throw std::invalid_argument("Embedding::backward expects gradOutput shape [batchSize, sequenceLength, embeddingDim].");
    }

    size_t batchSize = gradOutput.shape()[0];
    size_t sequenceLength = gradOutput.shape()[1];
    size_t gradEmbeddingDim = gradOutput.shape()[2];

    if (gradEmbeddingDim != embeddingDim_) {
        throw std::invalid_argument("Embedding::backward embedding dimension mismatch.");
    }

    if (
        cachedTokenIds_.shape()[0] != batchSize ||
        cachedTokenIds_.shape()[1] != sequenceLength
        ) {
        throw std::invalid_argument("Embedding::backward cached token shape mismatch.");
    }

    table_.grad.fill(0.0f);

    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t t = 0; t < sequenceLength; ++t) {
            int tokenId = static_cast<int>(cachedTokenIds_.at({ b, t }));

            if (tokenId < 0 || static_cast<size_t>(tokenId) >= vocabSize_) {
                throw std::out_of_range("Embedding cached token ID out of range.");
            }

            for (size_t e = 0; e < embeddingDim_; ++e) {
                table_.grad.at({ static_cast<size_t>(tokenId), e }) +=
                    gradOutput.at({ b, t, e });
            }
        }
    }

    return Tensor(cachedTokenIds_.shape(), 0.0f);
}

const Tensor& Embedding::table() const {
    return table_.value;
}

std::vector<Parameter*> Embedding::parameters() {
    return { &table_ };
}