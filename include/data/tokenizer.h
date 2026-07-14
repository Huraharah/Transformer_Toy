#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <ostream>


// ============================================================
// Tokenizer configuration
// ============================================================

enum class TokenizerType {
    Character,
    Word,
    BPE
};

std::ostream& operator<<(
    std::ostream& output,
    TokenizerType type
    );

std::string tokenizerTypeToString(
    TokenizerType type
);

struct TokenizerConfig {
    TokenizerType type = TokenizerType::Character;

    // Primarily used by trainable vocabulary tokenizers.
    std::size_t vocabSize = 512;

    // Minimum frequency required for vocabulary entries or merges.
    std::size_t minFrequency = 2;

    // Optional path for loading or saving tokenizer state.
    std::string modelPath;

    // Train a tokenizer if modelPath does not exist or is empty.
    bool trainIfMissing = true;

    // Word tokenizer behavior.
    bool preserveWhitespace = true;
    bool preservePunctuation = true;
};


// ============================================================
// Base tokenizer interface
// ============================================================

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual std::string name() const = 0;
    virtual TokenizerType type() const = 0;

    virtual void train(
        const std::string& corpus
    ) = 0;

    virtual std::vector<int> encode(
        const std::string& text
    ) const = 0;

    virtual std::string decode(
        const std::vector<int>& tokenIds
    ) const = 0;

    virtual std::size_t vocabSize() const = 0;

    virtual bool save(
        const std::string& path
    ) const = 0;

    virtual bool load(
        const std::string& path
    ) = 0;

    virtual bool isTrained() const = 0;

};


// ============================================================
// Character tokenizer
// ============================================================

class CharTokenizer final : public Tokenizer {
private:
    std::vector<char> idToChar_;
    std::unordered_map<char, int> charToId_;

public:
    CharTokenizer() = default;

    std::string name() const override;
    TokenizerType type() const override;

    void train(
        const std::string& corpus
    ) override;

    // Retained as a descriptive compatibility helper.
    void buildFromText(
        const std::string& text
    );

    std::vector<int> encode(
        const std::string& text
    ) const override;

    std::string decode(
        const std::vector<int>& tokenIds
    ) const override;

    std::size_t vocabSize() const override;

    bool save(
        const std::string& path
    ) const override;

    bool load(
        const std::string& path
    ) override;

    bool isTrained() const override;

    char idToChar(
        int tokenId
    ) const;

    int charToId(
        char character
    ) const;
};


// ============================================================
// Word tokenizer
// ============================================================

class WordTokenizer final : public Tokenizer {
private:
    std::size_t targetVocabSize_;
    std::size_t minFrequency_;

    bool preserveWhitespace_;
    bool preservePunctuation_;

    std::vector<std::string> idToToken_;
    std::unordered_map<std::string, int> tokenToId_;

    std::vector<std::string> splitText(
        const std::string& text
    ) const;

public:
    explicit WordTokenizer(
        std::size_t targetVocabSize = 0,
        std::size_t minFrequency = 1,
        bool preserveWhitespace = true,
        bool preservePunctuation = true
    );

    std::string name() const override;
    TokenizerType type() const override;

    void train(
        const std::string& corpus
    ) override;

    std::vector<int> encode(
        const std::string& text
    ) const override;

    std::string decode(
        const std::vector<int>& tokenIds
    ) const override;

    std::size_t vocabSize() const override;

    bool save(
        const std::string& path
    ) const override;

    bool load(
        const std::string& path
    ) override;

    bool isTrained() const override;

    const std::string& idToToken(
        int tokenId
    ) const;

    int tokenToId(
        const std::string& token
    ) const;
};


// ============================================================
// Abstract subword tokenizer
//
// BPE is a subword tokenizer. This base class stores the common
// vocabulary representation required by BPE and possible future
// algorithms such as unigram tokenization.
// ============================================================

class SubwordTokenizer : public Tokenizer {
protected:
    std::size_t targetVocabSize_;
    std::size_t minFrequency_;

    std::vector<std::string> idToToken_;
    std::unordered_map<std::string, int> tokenToId_;

    explicit SubwordTokenizer(
        std::size_t targetVocabSize,
        std::size_t minFrequency
    );

public:
    ~SubwordTokenizer() override = default;

    std::size_t vocabSize() const override;
    bool isTrained() const override;

    const std::string& idToToken(
        int tokenId
    ) const;

    int tokenToId(
        const std::string& token
    ) const;
};


// ============================================================
// Byte Pair Encoding tokenizer
// ============================================================

class BPETokenizer final : public SubwordTokenizer {
public:
    using MergePair =
        std::pair<std::string, std::string>;

private:
    std::vector<MergePair> merges_;

    std::unordered_map<
        std::string,
        std::size_t
    > mergeRanks_;

    std::vector<std::string> initializeSymbols(
        const std::string& text
    ) const;

    std::vector<std::string> applyMerges(
        std::vector<std::string> symbols
    ) const;

    static std::string makeMergeKey(
        const std::string& left,
        const std::string& right
    );

public:
    explicit BPETokenizer(
        std::size_t targetVocabSize = 512,
        std::size_t minFrequency = 2
    );

    std::string name() const override;
    TokenizerType type() const override;

    void train(
        const std::string& corpus
    ) override;

    std::vector<int> encode(
        const std::string& text
    ) const override;

    std::string decode(
        const std::vector<int>& tokenIds
    ) const override;

    bool save(
        const std::string& path
    ) const override;

    bool load(
        const std::string& path
    ) override;

    const std::vector<MergePair>& merges() const;
};


// ============================================================
// Tokenizer factory
// ============================================================

std::unique_ptr<Tokenizer> createTokenizer(
    const TokenizerConfig& config
);