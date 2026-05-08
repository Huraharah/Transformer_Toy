#include "data/tokenizer.h"

#include <set>
#include <stdexcept>

CharTokenizer::CharTokenizer() = default;

void CharTokenizer::buildFromText(const std::string& text) {
    std::set<char> uniqueChars(text.begin(), text.end());

    idToChar_.clear();
    charToId_.clear();

    for (char c : uniqueChars) {
        size_t id = idToChar_.size();
        idToChar_.push_back(c);
        charToId_[c] = id;
    }
}

std::vector<size_t> CharTokenizer::encode(const std::string& text) const {
    std::vector<size_t> ids;
    ids.reserve(text.size());

    for (char c : text) {
        auto it = charToId_.find(c);

        if (it == charToId_.end()) {
            throw std::out_of_range("Character not found in tokenizer vocabulary.");
        }

        ids.push_back(it->second);
    }

    return ids;
}

std::string CharTokenizer::decode(const std::vector<size_t>& ids) const {
    std::string text;
    text.reserve(ids.size());

    for (size_t id : ids) {
        if (id >= idToChar_.size()) {
            throw std::out_of_range("Token ID out of tokenizer vocabulary range.");
        }

        text.push_back(idToChar_[id]);
    }

    return text;
}

size_t CharTokenizer::vocabSize() const {
    return idToChar_.size();
}

char CharTokenizer::idToChar(size_t id) const {
    if (id >= idToChar_.size()) {
        throw std::out_of_range("Token ID out of tokenizer vocabulary range.");
    }

    return idToChar_[id];
}

size_t CharTokenizer::charToId(char c) const {
    auto it = charToId_.find(c);

    if (it == charToId_.end()) {
        throw std::out_of_range("Character not found in tokenizer vocabulary.");
    }

    return it->second;
}