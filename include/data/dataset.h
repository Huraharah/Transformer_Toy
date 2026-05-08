#pragma once

#include <string>
#include <vector>

#include "core/tensor.h"
#include "data/tokenizer.h"
#include "core/random.h"

class TextDataset {
private:
    std::string rawText_;
    std::vector<size_t> tokenIds_;

    CharTokenizer tokenizer_;
    size_t contextLength_;

public:
    TextDataset(const std::string& filePath, size_t contextLength);

    Tensor getInputWindow(size_t startIndex) const;
    Tensor getTargetWindow(size_t startIndex) const;
    void getBatch(size_t batchSize, Random& rng, Tensor& inputs, Tensor& targets) const;

    size_t numWindows() const;
    size_t contextLength() const;
    size_t vocabSize() const;

    const CharTokenizer& tokenizer() const;
    const std::string& rawText() const;
};