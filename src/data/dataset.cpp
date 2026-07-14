#include "data/dataset.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>


TextDataset::TextDataset(
    const std::string& filePath,
    std::size_t contextLength
)
    : TextDataset(
        filePath,
        contextLength,
        TokenizerConfig{}
    ) {
}


TextDataset::TextDataset(
    const std::string& filePath,
    std::size_t contextLength,
    const TokenizerConfig& tokenizerConfig
)
    : contextLength_(contextLength) {

    if (contextLength_ == 0) {
        throw std::invalid_argument(
            "TextDataset context length must be greater than zero."
        );
    }

    loadText(filePath);
    initializeTokenizer(tokenizerConfig);

    tokenIds_ = tokenizer_->encode(rawText_);

    if (tokenIds_.size() <= contextLength_) {
        throw std::runtime_error(
            "Dataset is too small for the selected context length."
        );
    }
}


void TextDataset::loadText(
    const std::string& filePath
) {
    std::ifstream file(
        filePath,
        std::ios::binary
    );

    if (!file) {
        throw std::runtime_error(
            "Failed to open dataset file: " +
            filePath
        );
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();

    if (!file.good() && !file.eof()) {
        throw std::runtime_error(
            "Failed while reading dataset file: " +
            filePath
        );
    }

    rawText_ = buffer.str();

    if (rawText_.empty()) {
        throw std::runtime_error(
            "Dataset file is empty: " +
            filePath
        );
    }
}


void TextDataset::initializeTokenizer(
    const TokenizerConfig& tokenizerConfig
) {
    tokenizer_ = createTokenizer(
        tokenizerConfig
    );

    if (!tokenizer_) {
        throw std::runtime_error(
            "Tokenizer factory returned a null tokenizer."
        );
    }

    bool loadedTokenizer = false;

    /*
        If a tokenizer model path was provided and the file exists,
        first attempt to load it.

        This is particularly important for BPE because the exact merge
        order must remain identical between training and inference.
    */
    if (
        !tokenizerConfig.modelPath.empty() &&
        std::filesystem::exists(
            tokenizerConfig.modelPath
        )
        ) {
        loadedTokenizer = tokenizer_->load(
            tokenizerConfig.modelPath
        );

        if (!loadedTokenizer) {
            throw std::runtime_error(
                "Failed to load tokenizer model: " +
                tokenizerConfig.modelPath
            );
        }
    }

    if (!loadedTokenizer) {
        if (!tokenizerConfig.trainIfMissing) {
            throw std::runtime_error(
                "Tokenizer model was unavailable and "
                "trainIfMissing is false."
            );
        }

        tokenizer_->train(rawText_);

        if (!tokenizer_->isTrained()) {
            throw std::runtime_error(
                "Tokenizer training completed without producing "
                "a valid vocabulary."
            );
        }

        /*
            Save newly trained tokenizer state when a model path was
            supplied.
        */
        if (!tokenizerConfig.modelPath.empty()) {
            const std::filesystem::path tokenizerPath(
                tokenizerConfig.modelPath
            );

            if (
                tokenizerPath.has_parent_path() &&
                !tokenizerPath.parent_path().empty()
                ) {
                std::filesystem::create_directories(
                    tokenizerPath.parent_path()
                );
            }

            if (
                !tokenizer_->save(
                    tokenizerConfig.modelPath
                )
                ) {
                throw std::runtime_error(
                    "Failed to save trained tokenizer model: " +
                    tokenizerConfig.modelPath
                );
            }
        }
    }
}


Tensor TextDataset::getInputWindow(
    std::size_t startIndex
) const {
    if (
        startIndex + contextLength_ >=
        tokenIds_.size()
        ) {
        throw std::out_of_range(
            "Input window start index is out of range."
        );
    }

    Tensor input(
        { 1, contextLength_ },
        0.0f
    );

    for (
        std::size_t position = 0;
        position < contextLength_;
        ++position
        ) {
        input.at({ 0, position }) =
            static_cast<float>(
                tokenIds_[
                    startIndex + position
                ]
                );
    }

    return input;
}


Tensor TextDataset::getTargetWindow(
    std::size_t startIndex
) const {
    if (
        startIndex + contextLength_ >=
        tokenIds_.size()
        ) {
        throw std::out_of_range(
            "Target window start index is out of range."
        );
    }

    Tensor target(
        { 1, contextLength_ },
        0.0f
    );

    for (
        std::size_t position = 0;
        position < contextLength_;
        ++position
        ) {
        target.at({ 0, position }) =
            static_cast<float>(
                tokenIds_[
                    startIndex +
                        position +
                        1
                ]
                );
    }

    return target;
}


void TextDataset::getBatch(
    std::size_t batchSize,
    Random& rng,
    Tensor& inputs,
    Tensor& targets
) const {
    if (batchSize == 0) {
        throw std::invalid_argument(
            "TextDataset batch size must be greater than zero."
        );
    }

    inputs = Tensor(
        { batchSize, contextLength_ },
        0.0f
    );

    targets = Tensor(
        { batchSize, contextLength_ },
        0.0f
    );

    const std::size_t windowCount =
        numWindows();

    if (windowCount == 0) {
        throw std::runtime_error(
            "TextDataset contains no valid training windows."
        );
    }

    for (
        std::size_t batchIndex = 0;
        batchIndex < batchSize;
        ++batchIndex
        ) {
        const int startIndex =
            rng.randint(
                0,
                static_cast<int>(
                    windowCount - 1
                    )
            );

        for (
            std::size_t position = 0;
            position < contextLength_;
            ++position
            ) {
            const std::size_t sourceIndex =
                static_cast<std::size_t>(
                    startIndex
                    ) +
                position;

            inputs.at({
                batchIndex,
                position
                }) = static_cast<float>(
                    tokenIds_[sourceIndex]
                    );

            targets.at({
                batchIndex,
                position
                }) = static_cast<float>(
                    tokenIds_[sourceIndex + 1]
                    );
        }
    }
}


std::size_t TextDataset::numWindows() const {
    if (
        tokenIds_.size() <=
        contextLength_
        ) {
        return 0;
    }

    return
        tokenIds_.size() -
        contextLength_;
}


std::size_t TextDataset::contextLength() const {
    return contextLength_;
}


std::size_t TextDataset::vocabSize() const {
    if (!tokenizer_) {
        return 0;
    }

    return tokenizer_->vocabSize();
}


std::size_t TextDataset::tokenCount() const {
    return tokenIds_.size();
}


const Tokenizer& TextDataset::tokenizer() const {
    if (!tokenizer_) {
        throw std::runtime_error(
            "TextDataset does not own an initialized tokenizer."
        );
    }

    return *tokenizer_;
}


Tokenizer& TextDataset::tokenizer() {
    if (!tokenizer_) {
        throw std::runtime_error(
            "TextDataset does not own an initialized tokenizer."
        );
    }

    return *tokenizer_;
}


const std::vector<int>&
TextDataset::tokenIds() const {
    return tokenIds_;
}


const std::string&
TextDataset::rawText() const {
    return rawText_;
}