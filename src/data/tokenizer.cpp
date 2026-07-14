#include "data/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

std::string tokenizerTypeToString(
    TokenizerType type
) {
    switch (type) {
    case TokenizerType::Character:
        return "Character";

    case TokenizerType::Word:
        return "Word";

    case TokenizerType::BPE:
        return "BPE";

    default:
        return "Unknown";
    }
}

namespace {

    constexpr int INVALID_TOKEN_ID = -1;

    void validateTokenId(
        int tokenId,
        std::size_t vocabularySize,
        const std::string& tokenizerName
    ) {
        if (
            tokenId < 0 ||
            static_cast<std::size_t>(tokenId) >= vocabularySize
            ) {
            throw std::out_of_range(
                tokenizerName +
                ": token ID is outside the vocabulary range."
            );
        }
    }

    void writeSizedString(
        std::ostream& output,
        const std::string& value
    ) {
        output << value.size() << '\n';
        output.write(
            value.data(),
            static_cast<std::streamsize>(value.size())
        );
        output << '\n';
    }

    std::string readSizedString(
        std::istream& input
    ) {
        std::size_t size = 0;

        if (!(input >> size)) {
            throw std::runtime_error(
                "Tokenizer load failed while reading string size."
            );
        }

        input.ignore(
            std::numeric_limits<std::streamsize>::max(),
            '\n'
        );

        std::string value(size, '\0');

        if (size > 0) {
            input.read(
                value.data(),
                static_cast<std::streamsize>(size)
            );

            if (!input) {
                throw std::runtime_error(
                    "Tokenizer load failed while reading string data."
                );
            }
        }

        if (input.peek() == '\n') {
            input.get();
        }

        return value;
    }

} // namespace

std::ostream& operator<<(
    std::ostream& output,
    TokenizerType type
    ) {
    switch (type) {
    case TokenizerType::Character:
        output << "Character";
        break;

    case TokenizerType::Word:
        output << "Word";
        break;

    case TokenizerType::BPE:
        output << "BPE";
        break;

    default:
        output << "Unknown";
        break;
    }

    return output;
}


// ============================================================
// Character tokenizer
// ============================================================

std::string CharTokenizer::name() const {
    return "character";
}

TokenizerType CharTokenizer::type() const {
    return TokenizerType::Character;
}

void CharTokenizer::train(
    const std::string& corpus
) {
    buildFromText(corpus);
}

void CharTokenizer::buildFromText(
    const std::string& text
) {
    std::set<char> uniqueChars(
        text.begin(),
        text.end()
    );

    idToChar_.clear();
    charToId_.clear();

    idToChar_.reserve(uniqueChars.size());

    for (char character : uniqueChars) {
        const int tokenId =
            static_cast<int>(idToChar_.size());

        idToChar_.push_back(character);
        charToId_[character] = tokenId;
    }
}

std::vector<int> CharTokenizer::encode(
    const std::string& text
) const {
    if (!isTrained()) {
        throw std::runtime_error(
            "CharTokenizer::encode called before training or loading."
        );
    }

    std::vector<int> tokenIds;
    tokenIds.reserve(text.size());

    for (char character : text) {
        const auto iterator =
            charToId_.find(character);

        if (iterator == charToId_.end()) {
            throw std::out_of_range(
                "CharTokenizer: character not found in vocabulary."
            );
        }

        tokenIds.push_back(iterator->second);
    }

    return tokenIds;
}

std::string CharTokenizer::decode(
    const std::vector<int>& tokenIds
) const {
    if (!isTrained()) {
        throw std::runtime_error(
            "CharTokenizer::decode called before training or loading."
        );
    }

    std::string text;
    text.reserve(tokenIds.size());

    for (int tokenId : tokenIds) {
        validateTokenId(
            tokenId,
            idToChar_.size(),
            "CharTokenizer"
        );

        text.push_back(
            idToChar_[static_cast<std::size_t>(tokenId)]
        );
    }

    return text;
}

std::size_t CharTokenizer::vocabSize() const {
    return idToChar_.size();
}

bool CharTokenizer::save(
    const std::string& path
) const {
    if (!isTrained()) {
        return false;
    }

    std::ofstream output(
        path,
        std::ios::binary
    );

    if (!output) {
        return false;
    }

    output << "CHAR_TOKENIZER_V1\n";
    output << idToChar_.size() << '\n';

    for (char character : idToChar_) {
        const unsigned int byteValue =
            static_cast<unsigned char>(character);

        output << byteValue << '\n';
    }

    return static_cast<bool>(output);
}

bool CharTokenizer::load(
    const std::string& path
) {
    std::ifstream input(
        path,
        std::ios::binary
    );

    if (!input) {
        return false;
    }

    std::string header;
    std::getline(input, header);

    if (header != "CHAR_TOKENIZER_V1") {
        return false;
    }

    std::size_t vocabularySize = 0;

    if (!(input >> vocabularySize)) {
        return false;
    }

    std::vector<char> newIdToChar;
    std::unordered_map<char, int> newCharToId;

    newIdToChar.reserve(vocabularySize);

    for (
        std::size_t tokenIndex = 0;
        tokenIndex < vocabularySize;
        ++tokenIndex
        ) {
        unsigned int byteValue = 0;

        if (!(input >> byteValue)) {
            return false;
        }

        if (
            byteValue >
            std::numeric_limits<unsigned char>::max()
            ) {
            return false;
        }

        const char character =
            static_cast<char>(
                static_cast<unsigned char>(byteValue)
                );

        const int tokenId =
            static_cast<int>(newIdToChar.size());

        newIdToChar.push_back(character);
        newCharToId[character] = tokenId;
    }

    idToChar_ = std::move(newIdToChar);
    charToId_ = std::move(newCharToId);

    return true;
}

bool CharTokenizer::isTrained() const {
    return !idToChar_.empty();
}

char CharTokenizer::idToChar(
    int tokenId
) const {
    validateTokenId(
        tokenId,
        idToChar_.size(),
        "CharTokenizer"
    );

    return idToChar_[
        static_cast<std::size_t>(tokenId)
    ];
}

int CharTokenizer::charToId(
    char character
) const {
    const auto iterator =
        charToId_.find(character);

    if (iterator == charToId_.end()) {
        throw std::out_of_range(
            "CharTokenizer: character not found in vocabulary."
        );
    }

    return iterator->second;
}


// ============================================================
// Word tokenizer
// ============================================================

WordTokenizer::WordTokenizer(
    std::size_t targetVocabSize,
    std::size_t minFrequency,
    bool preserveWhitespace,
    bool preservePunctuation
)
    : targetVocabSize_(targetVocabSize),
    minFrequency_(minFrequency),
    preserveWhitespace_(preserveWhitespace),
    preservePunctuation_(preservePunctuation) {

    if (minFrequency_ == 0) {
        throw std::invalid_argument(
            "WordTokenizer minFrequency must be greater than zero."
        );
    }
}

std::string WordTokenizer::name() const {
    return "word";
}

TokenizerType WordTokenizer::type() const {
    return TokenizerType::Word;
}

std::vector<std::string> WordTokenizer::splitText(
    const std::string& text
) const {
    std::vector<std::string> tokens;
    std::string currentToken;

    enum class CharacterCategory {
        Word,
        Whitespace,
        Punctuation
    };

    auto categoryOf = [](
        unsigned char character
        ) -> CharacterCategory {
            if (std::isspace(character)) {
                return CharacterCategory::Whitespace;
            }

            if (
                std::isalnum(character) ||
                character == '_' ||
                character == '\''
                ) {
                return CharacterCategory::Word;
            }

            return CharacterCategory::Punctuation;
        };

    auto flushCurrent = [&]() {
        if (!currentToken.empty()) {
            tokens.push_back(currentToken);
            currentToken.clear();
        }
        };

    CharacterCategory currentCategory =
        CharacterCategory::Word;

    bool hasCurrentCategory = false;

    for (char rawCharacter : text) {
        const unsigned char character =
            static_cast<unsigned char>(rawCharacter);

        const CharacterCategory newCategory =
            categoryOf(character);

        if (
            newCategory == CharacterCategory::Whitespace &&
            !preserveWhitespace_
            ) {
            flushCurrent();
            hasCurrentCategory = false;
            continue;
        }

        if (
            newCategory == CharacterCategory::Punctuation &&
            !preservePunctuation_
            ) {
            flushCurrent();
            hasCurrentCategory = false;
            continue;
        }

        if (
            !hasCurrentCategory ||
            newCategory == currentCategory
            ) {
            currentToken.push_back(rawCharacter);
            currentCategory = newCategory;
            hasCurrentCategory = true;
        }
        else {
            flushCurrent();

            currentToken.push_back(rawCharacter);
            currentCategory = newCategory;
            hasCurrentCategory = true;
        }
    }

    flushCurrent();

    return tokens;
}

void WordTokenizer::train(
    const std::string& corpus
) {
    if (corpus.empty()) {
        throw std::invalid_argument(
            "WordTokenizer cannot train on an empty corpus."
        );
    }

    const std::vector<std::string> corpusTokens =
        splitText(corpus);

    std::unordered_map<std::string, std::size_t>
        frequencies;

    for (const std::string& token : corpusTokens) {
        ++frequencies[token];
    }

    std::vector<
        std::pair<std::string, std::size_t>
    > sortedTokens;

    sortedTokens.reserve(frequencies.size());

    for (const auto& entry : frequencies) {
        if (entry.second >= minFrequency_) {
            sortedTokens.push_back(entry);
        }
    }

    std::sort(
        sortedTokens.begin(),
        sortedTokens.end(),
        [](
            const auto& left,
            const auto& right
            ) {
                if (left.second != right.second) {
                    return left.second > right.second;
                }

                return left.first < right.first;
        }
    );

    if (
        targetVocabSize_ > 0 &&
        sortedTokens.size() > targetVocabSize_
        ) {
        sortedTokens.resize(targetVocabSize_);
    }

    idToToken_.clear();
    tokenToId_.clear();

    idToToken_.reserve(sortedTokens.size());

    for (const auto& entry : sortedTokens) {
        const int tokenId =
            static_cast<int>(idToToken_.size());

        idToToken_.push_back(entry.first);
        tokenToId_[entry.first] = tokenId;
    }
}

std::vector<int> WordTokenizer::encode(
    const std::string& text
) const {
    if (!isTrained()) {
        throw std::runtime_error(
            "WordTokenizer::encode called before training or loading."
        );
    }

    const std::vector<std::string> splitTokens =
        splitText(text);

    std::vector<int> tokenIds;
    tokenIds.reserve(splitTokens.size());

    for (const std::string& token : splitTokens) {
        const auto iterator =
            tokenToId_.find(token);

        if (iterator == tokenToId_.end()) {
            throw std::out_of_range(
                "WordTokenizer: token not found in vocabulary: " +
                token
            );
        }

        tokenIds.push_back(iterator->second);
    }

    return tokenIds;
}

std::string WordTokenizer::decode(
    const std::vector<int>& tokenIds
) const {
    if (!isTrained()) {
        throw std::runtime_error(
            "WordTokenizer::decode called before training or loading."
        );
    }

    std::string text;

    for (int tokenId : tokenIds) {
        validateTokenId(
            tokenId,
            idToToken_.size(),
            "WordTokenizer"
        );

        text += idToToken_[
            static_cast<std::size_t>(tokenId)
        ];
    }

    return text;
}

std::size_t WordTokenizer::vocabSize() const {
    return idToToken_.size();
}

bool WordTokenizer::save(
    const std::string& path
) const {
    if (!isTrained()) {
        return false;
    }

    std::ofstream output(
        path,
        std::ios::binary
    );

    if (!output) {
        return false;
    }

    output << "WORD_TOKENIZER_V1\n";
    output << targetVocabSize_ << '\n';
    output << minFrequency_ << '\n';
    output << preserveWhitespace_ << '\n';
    output << preservePunctuation_ << '\n';
    output << idToToken_.size() << '\n';

    for (const std::string& token : idToToken_) {
        writeSizedString(output, token);
    }

    return static_cast<bool>(output);
}

bool WordTokenizer::load(
    const std::string& path
) {
    std::ifstream input(
        path,
        std::ios::binary
    );

    if (!input) {
        return false;
    }

    std::string header;
    std::getline(input, header);

    if (header != "WORD_TOKENIZER_V1") {
        return false;
    }

    std::size_t vocabularySize = 0;

    if (
        !(input >> targetVocabSize_) ||
        !(input >> minFrequency_) ||
        !(input >> preserveWhitespace_) ||
        !(input >> preservePunctuation_) ||
        !(input >> vocabularySize)
        ) {
        return false;
    }

    input.ignore(
        std::numeric_limits<std::streamsize>::max(),
        '\n'
    );

    std::vector<std::string> newIdToToken;
    std::unordered_map<std::string, int>
        newTokenToId;

    newIdToToken.reserve(vocabularySize);

    try {
        for (
            std::size_t tokenIndex = 0;
            tokenIndex < vocabularySize;
            ++tokenIndex
            ) {
            std::string token =
                readSizedString(input);

            const int tokenId =
                static_cast<int>(newIdToToken.size());

            newIdToToken.push_back(token);
            newTokenToId[token] = tokenId;
        }
    }
    catch (...) {
        return false;
    }

    idToToken_ = std::move(newIdToToken);
    tokenToId_ = std::move(newTokenToId);

    return true;
}

bool WordTokenizer::isTrained() const {
    return !idToToken_.empty();
}

const std::string& WordTokenizer::idToToken(
    int tokenId
) const {
    validateTokenId(
        tokenId,
        idToToken_.size(),
        "WordTokenizer"
    );

    return idToToken_[
        static_cast<std::size_t>(tokenId)
    ];
}

int WordTokenizer::tokenToId(
    const std::string& token
) const {
    const auto iterator =
        tokenToId_.find(token);

    if (iterator == tokenToId_.end()) {
        return INVALID_TOKEN_ID;
    }

    return iterator->second;
}


// ============================================================
// Shared subword-tokenizer functionality
// ============================================================

SubwordTokenizer::SubwordTokenizer(
    std::size_t targetVocabSize,
    std::size_t minFrequency
)
    : targetVocabSize_(targetVocabSize),
    minFrequency_(minFrequency) {

    if (targetVocabSize_ == 0) {
        throw std::invalid_argument(
            "SubwordTokenizer targetVocabSize must be greater than zero."
        );
    }

    if (minFrequency_ == 0) {
        throw std::invalid_argument(
            "SubwordTokenizer minFrequency must be greater than zero."
        );
    }
}

std::size_t SubwordTokenizer::vocabSize() const {
    return idToToken_.size();
}

bool SubwordTokenizer::isTrained() const {
    return !idToToken_.empty();
}

const std::string& SubwordTokenizer::idToToken(
    int tokenId
) const {
    validateTokenId(
        tokenId,
        idToToken_.size(),
        "SubwordTokenizer"
    );

    return idToToken_[
        static_cast<std::size_t>(tokenId)
    ];
}

int SubwordTokenizer::tokenToId(
    const std::string& token
) const {
    const auto iterator =
        tokenToId_.find(token);

    if (iterator == tokenToId_.end()) {
        return INVALID_TOKEN_ID;
    }

    return iterator->second;
}


// ============================================================
// BPE tokenizer
// ============================================================

BPETokenizer::BPETokenizer(
    std::size_t targetVocabSize,
    std::size_t minFrequency
)
    : SubwordTokenizer(
        targetVocabSize,
        minFrequency
    ) {
}

std::string BPETokenizer::name() const {
    return "bpe";
}

TokenizerType BPETokenizer::type() const {
    return TokenizerType::BPE;
}

std::string BPETokenizer::makeMergeKey(
    const std::string& left,
    const std::string& right
) {
    /*
        Length-prefixing avoids collisions such as:

            ("ab", "c")
            ("a", "bc")

        which would collide if simply concatenated.
    */
    return
        std::to_string(left.size()) +
        ":" +
        left +
        std::to_string(right.size()) +
        ":" +
        right;
}

std::vector<std::string>
BPETokenizer::initializeSymbols(
    const std::string& text
) const {
    std::vector<std::string> symbols;
    symbols.reserve(text.size());

    for (unsigned char byte : text) {
        symbols.emplace_back(
            1,
            static_cast<char>(byte)
        );
    }

    return symbols;
}

std::vector<std::string> BPETokenizer::applyMerges(
    std::vector<std::string> symbols
) const {
    if (symbols.size() < 2 || merges_.empty()) {
        return symbols;
    }

    while (true) {
        std::size_t bestRank =
            std::numeric_limits<std::size_t>::max();

        std::string bestLeft;
        std::string bestRight;

        bool foundMerge = false;

        for (
            std::size_t index = 0;
            index + 1 < symbols.size();
            ++index
            ) {
            const std::string key =
                makeMergeKey(
                    symbols[index],
                    symbols[index + 1]
                );

            const auto iterator =
                mergeRanks_.find(key);

            if (
                iterator != mergeRanks_.end() &&
                iterator->second < bestRank
                ) {
                bestRank = iterator->second;
                bestLeft = symbols[index];
                bestRight = symbols[index + 1];
                foundMerge = true;
            }
        }

        if (!foundMerge) {
            break;
        }

        std::vector<std::string> mergedSymbols;
        mergedSymbols.reserve(symbols.size());

        for (
            std::size_t index = 0;
            index < symbols.size();
            ) {
            if (
                index + 1 < symbols.size() &&
                symbols[index] == bestLeft &&
                symbols[index + 1] == bestRight
                ) {
                mergedSymbols.push_back(
                    bestLeft + bestRight
                );

                index += 2;
            }
            else {
                mergedSymbols.push_back(
                    symbols[index]
                );

                ++index;
            }
        }

        symbols = std::move(mergedSymbols);
    }

    return symbols;
}

void BPETokenizer::train(
    const std::string& corpus
) {
    if (corpus.empty()) {
        throw std::invalid_argument(
            "BPETokenizer cannot train on an empty corpus."
        );
    }

    std::vector<std::string> symbols =
        initializeSymbols(corpus);

    std::set<std::string> vocabularySet(
        symbols.begin(),
        symbols.end()
    );

    if (
        targetVocabSize_ <
        vocabularySet.size()
        ) {
        throw std::invalid_argument(
            "BPETokenizer target vocabulary is smaller "
            "than the initial character vocabulary."
        );
    }

    merges_.clear();
    mergeRanks_.clear();

    while (
        vocabularySet.size() < targetVocabSize_ &&
        symbols.size() >= 2
        ) {
        using Pair =
            std::pair<std::string, std::string>;

        std::map<Pair, std::size_t> pairFrequencies;

        for (
            std::size_t index = 0;
            index + 1 < symbols.size();
            ++index
            ) {
            const Pair pair = {
                symbols[index],
                symbols[index + 1]
            };

            ++pairFrequencies[pair];
        }

        if (pairFrequencies.empty()) {
            break;
        }

        auto bestIterator =
            pairFrequencies.begin();

        for (
            auto iterator = pairFrequencies.begin();
            iterator != pairFrequencies.end();
            ++iterator
            ) {
            if (
                iterator->second >
                bestIterator->second
                ) {
                bestIterator = iterator;
            }
            else if (
                iterator->second ==
                bestIterator->second &&
                iterator->first <
                bestIterator->first
                ) {
                /*
                    Deterministic lexical tie-breaking makes BPE
                    training reproducible.
                */
                bestIterator = iterator;
            }
        }

        if (
            bestIterator->second <
            minFrequency_
            ) {
            break;
        }

        const std::string& left =
            bestIterator->first.first;

        const std::string& right =
            bestIterator->first.second;

        const std::string mergedToken =
            left + right;

        const std::size_t mergeRank =
            merges_.size();

        merges_.push_back({
            left,
            right
            });

        mergeRanks_[
            makeMergeKey(left, right)
        ] = mergeRank;

        vocabularySet.insert(mergedToken);

        std::vector<std::string> nextSymbols;
        nextSymbols.reserve(symbols.size());

        for (
            std::size_t index = 0;
            index < symbols.size();
            ) {
            if (
                index + 1 < symbols.size() &&
                symbols[index] == left &&
                symbols[index + 1] == right
                ) {
                nextSymbols.push_back(mergedToken);
                index += 2;
            }
            else {
                nextSymbols.push_back(
                    symbols[index]
                );

                ++index;
            }
        }

        symbols = std::move(nextSymbols);
    }

    /*
        Token IDs are assigned deterministically:

        1. Base single-character tokens in lexical byte order.
        2. Merged tokens in learned merge order.
    */
    std::set<std::string> baseTokens;

    for (unsigned char byte : corpus) {
        baseTokens.insert(
            std::string(
                1,
                static_cast<char>(byte)
            )
        );
    }

    idToToken_.clear();
    tokenToId_.clear();

    idToToken_.reserve(
        baseTokens.size() + merges_.size()
    );

    auto addToken = [&](const std::string& token) {
        if (tokenToId_.find(token) != tokenToId_.end()) {
            return;
        }

        const int tokenId =
            static_cast<int>(idToToken_.size());

        idToToken_.push_back(token);
        tokenToId_[token] = tokenId;
        };

    for (const std::string& token : baseTokens) {
        addToken(token);
    }

    for (const MergePair& merge : merges_) {
        addToken(
            merge.first + merge.second
        );
    }
}

std::vector<int> BPETokenizer::encode(
    const std::string& text
) const {
    if (!isTrained()) {
        throw std::runtime_error(
            "BPETokenizer::encode called before training or loading."
        );
    }

    std::vector<std::string> symbols =
        initializeSymbols(text);

    for (const std::string& symbol : symbols) {
        if (
            tokenToId_.find(symbol) ==
            tokenToId_.end()
            ) {
            throw std::out_of_range(
                "BPETokenizer: input contains a character "
                "not present in the trained vocabulary."
            );
        }
    }

    symbols = applyMerges(
        std::move(symbols)
    );

    std::vector<int> tokenIds;
    tokenIds.reserve(symbols.size());

    for (const std::string& token : symbols) {
        const auto iterator =
            tokenToId_.find(token);

        if (iterator == tokenToId_.end()) {
            throw std::logic_error(
                "BPETokenizer produced a merged token "
                "that is absent from its vocabulary."
            );
        }

        tokenIds.push_back(iterator->second);
    }

    return tokenIds;
}

std::string BPETokenizer::decode(
    const std::vector<int>& tokenIds
) const {
    if (!isTrained()) {
        throw std::runtime_error(
            "BPETokenizer::decode called before training or loading."
        );
    }

    std::string text;

    for (int tokenId : tokenIds) {
        validateTokenId(
            tokenId,
            idToToken_.size(),
            "BPETokenizer"
        );

        text += idToToken_[
            static_cast<std::size_t>(tokenId)
        ];
    }

    return text;
}

bool BPETokenizer::save(
    const std::string& path
) const {
    if (!isTrained()) {
        return false;
    }

    std::ofstream output(
        path,
        std::ios::binary
    );

    if (!output) {
        return false;
    }

    output << "BPE_TOKENIZER_V1\n";
    output << targetVocabSize_ << '\n';
    output << minFrequency_ << '\n';

    output << idToToken_.size() << '\n';

    for (const std::string& token : idToToken_) {
        writeSizedString(output, token);
    }

    output << merges_.size() << '\n';

    for (const MergePair& merge : merges_) {
        writeSizedString(output, merge.first);
        writeSizedString(output, merge.second);
    }

    return static_cast<bool>(output);
}

bool BPETokenizer::load(
    const std::string& path
) {
    std::ifstream input(
        path,
        std::ios::binary
    );

    if (!input) {
        return false;
    }

    std::string header;
    std::getline(input, header);

    if (header != "BPE_TOKENIZER_V1") {
        return false;
    }

    std::size_t vocabularySize = 0;
    std::size_t mergeCount = 0;

    if (
        !(input >> targetVocabSize_) ||
        !(input >> minFrequency_) ||
        !(input >> vocabularySize)
        ) {
        return false;
    }

    input.ignore(
        std::numeric_limits<std::streamsize>::max(),
        '\n'
    );

    std::vector<std::string> newIdToToken;
    std::unordered_map<std::string, int>
        newTokenToId;

    newIdToToken.reserve(vocabularySize);

    try {
        for (
            std::size_t tokenIndex = 0;
            tokenIndex < vocabularySize;
            ++tokenIndex
            ) {
            std::string token =
                readSizedString(input);

            const int tokenId =
                static_cast<int>(newIdToToken.size());

            newIdToToken.push_back(token);
            newTokenToId[token] = tokenId;
        }

        if (!(input >> mergeCount)) {
            return false;
        }

        input.ignore(
            std::numeric_limits<std::streamsize>::max(),
            '\n'
        );

        std::vector<MergePair> newMerges;
        std::unordered_map<
            std::string,
            std::size_t
        > newMergeRanks;

        newMerges.reserve(mergeCount);

        for (
            std::size_t mergeIndex = 0;
            mergeIndex < mergeCount;
            ++mergeIndex
            ) {
            std::string left =
                readSizedString(input);

            std::string right =
                readSizedString(input);

            newMergeRanks[
                makeMergeKey(left, right)
            ] = mergeIndex;

            newMerges.push_back({
                std::move(left),
                std::move(right)
                });
        }

        idToToken_ = std::move(newIdToToken);
        tokenToId_ = std::move(newTokenToId);
        merges_ = std::move(newMerges);
        mergeRanks_ = std::move(newMergeRanks);
    }
    catch (...) {
        return false;
    }

    return true;
}

const std::vector<BPETokenizer::MergePair>&
BPETokenizer::merges() const {
    return merges_;
}


// ============================================================
// Tokenizer factory
// ============================================================

std::unique_ptr<Tokenizer> createTokenizer(
    const TokenizerConfig& config
) {
    switch (config.type) {
    case TokenizerType::Character:
        return std::make_unique<
            CharTokenizer
        >();

    case TokenizerType::Word:
        return std::make_unique<
            WordTokenizer
        >(
            config.vocabSize,
            config.minFrequency,
            config.preserveWhitespace,
            config.preservePunctuation
        );

    case TokenizerType::BPE:
        return std::make_unique<
            BPETokenizer
        >(
            config.vocabSize,
            config.minFrequency
        );

    default:
        throw std::invalid_argument(
            "createTokenizer received an unsupported tokenizer type."
        );
    }
}