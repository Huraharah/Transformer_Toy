#include "data/dataset.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

TextDataset::TextDataset(const std::string& filePath, size_t contextLength)
    : contextLength_(contextLength) {

    std::ifstream file(filePath);

    if (!file) {
        throw std::runtime_error("Failed to open dataset file: " + filePath);
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    rawText_ = buffer.str();

    if (rawText_.empty()) {
        throw std::runtime_error("Dataset file is empty.");
    }

    tokenizer_.buildFromText(rawText_);
    tokenIds_ = tokenizer_.encode(rawText_);

    if (tokenIds_.size() <= contextLength_) {
        throw std::runtime_error("Dataset is too small for the selected context length.");
    }
}

Tensor TextDataset::getInputWindow(size_t startIndex) const {
    if (startIndex + contextLength_ >= tokenIds_.size()) {
        throw std::out_of_range("Input window start index out of range.");
    }

    Tensor input({ 1, contextLength_ }, 0.0f);

    for (size_t i = 0; i < contextLength_; ++i) {
        input.at({ 0, i }) = static_cast<float>(tokenIds_[startIndex + i]);
    }

    return input;
}

Tensor TextDataset::getTargetWindow(size_t startIndex) const {
    if (startIndex + contextLength_ >= tokenIds_.size()) {
        throw std::out_of_range("Target window start index out of range.");
    }

    Tensor target({ 1, contextLength_ }, 0.0f);

    for (size_t i = 0; i < contextLength_; ++i) {
        target.at({ 0, i }) = static_cast<float>(tokenIds_[startIndex + i + 1]);
    }

    return target;
}

void TextDataset::getBatch(
    size_t batchSize,
    Random& rng,
    Tensor& inputs,
    Tensor& targets
) const {
    inputs = Tensor({ batchSize, contextLength_ }, 0.0f);
    targets = Tensor({ batchSize, contextLength_ }, 0.0f);

    for (size_t b = 0; b < batchSize; ++b) {
        int startIndex = rng.randint(
            0,
            static_cast<int>(numWindows() - 1)
        );

        for (size_t t = 0; t < contextLength_; ++t) {
            inputs.at({ b, t }) =
                static_cast<float>(tokenIds_[startIndex + t]);

            targets.at({ b, t }) =
                static_cast<float>(tokenIds_[startIndex + t + 1]);
        }
    }
}

size_t TextDataset::numWindows() const {
    return tokenIds_.size() - contextLength_;
}

size_t TextDataset::contextLength() const {
    return contextLength_;
}

size_t TextDataset::vocabSize() const {
    return tokenizer_.vocabSize();
}

const CharTokenizer& TextDataset::tokenizer() const {
    return tokenizer_;
}

const std::string& TextDataset::rawText() const {
    return rawText_;
}