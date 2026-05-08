#pragma once

#include "core/tensor.h"
#include "core/random.h"

class Embedding {
private:
    size_t vocabSize_;
    size_t embeddingDim_;

    Tensor table_; // [vocabSize, embeddingDim]

public:
    Embedding(size_t vocabSize, size_t embeddingDim, Random& rng);

    Tensor forward(const Tensor& tokenIds) const;

    const Tensor& table() const;
};