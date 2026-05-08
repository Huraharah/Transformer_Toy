#include "layers/embedding.h"

#include <cmath>
#include <stdexcept>

Embedding::Embedding(size_t vocabSize, size_t embeddingDim, Random& rng)
    : vocabSize_(vocabSize),
    embeddingDim_(embeddingDim),
    table_({ vocabSize, embeddingDim }) {

    float limit = std::sqrt(6.0f / static_cast<float>(vocabSize + embeddingDim));

    for (size_t i = 0; i < table_.size(); ++i) {
        table_[i] = rng.uniform(-limit, limit);
    }
}

Tensor Embedding::forward(const Tensor& tokenIds) const {
    if (tokenIds.rank() != 2) {
        throw std::invalid_argument("Embedding::forward expects input shape [batchSize, sequenceLength].");
    }

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
                output.at({ b, t, e }) = table_.at({ static_cast<size_t>(tokenId), e });
            }
        }
    }

    return output;
}

const Tensor& Embedding::table() const {
    return table_;
}