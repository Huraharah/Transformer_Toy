#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "core/random.h"
#include "core/tensor.h"
#include "data/tokenizer.h"


class TextDataset {
private:
    std::string rawText_;
    std::vector<int> tokenIds_;

    std::unique_ptr<Tokenizer> tokenizer_;

    std::size_t contextLength_;

    void loadText(
        const std::string& filePath
    );

    void initializeTokenizer(
        const TokenizerConfig& tokenizerConfig
    );

public:
    /*
        Backward-compatible constructor.

        Uses the default TokenizerConfig, which currently selects
        character tokenization.
    */
    TextDataset(
        const std::string& filePath,
        std::size_t contextLength
    );

    /*
        Configurable tokenizer constructor.
    */
    TextDataset(
        const std::string& filePath,
        std::size_t contextLength,
        const TokenizerConfig& tokenizerConfig
    );

    /*
        TextDataset owns a unique_ptr and therefore should not be copied
        accidentally. Moving remains supported.
    */
    TextDataset(const TextDataset&) = delete;
    TextDataset& operator=(const TextDataset&) = delete;

    TextDataset(TextDataset&&) noexcept = default;
    TextDataset& operator=(TextDataset&&) noexcept = default;

    Tensor getInputWindow(
        std::size_t startIndex
    ) const;

    Tensor getTargetWindow(
        std::size_t startIndex
    ) const;

    void getBatch(
        std::size_t batchSize,
        Random& rng,
        Tensor& inputs,
        Tensor& targets
    ) const;

    std::size_t numWindows() const;
    std::size_t contextLength() const;
    std::size_t vocabSize() const;
    std::size_t tokenCount() const;

    const Tokenizer& tokenizer() const;
    Tokenizer& tokenizer();

    const std::vector<int>& tokenIds() const;
    const std::string& rawText() const;
};