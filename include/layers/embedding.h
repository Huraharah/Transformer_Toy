#pragma once

#include "core/tensor.h"
#include "core/random.h"
#include "core/parameter.h"

class Embedding {
private:
    size_t vocabSize_;
    size_t embeddingDim_;

    Tensor cachedTokenIds_;

    Parameter table_; // [vocabSize, embeddingDim]

public:
    Embedding(size_t vocabSize, size_t embeddingDim, Random& rng);

    Tensor forward(const Tensor& tokenIds);
	Tensor backward(const Tensor& gradOutput);

    const Tensor& table() const;
    std::vector<Parameter*> parameters();
};