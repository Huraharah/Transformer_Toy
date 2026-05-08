#pragma once

#include <string>
#include <vector>
#include <unordered_map>

class CharTokenizer {
private:
    std::vector<char> idToChar_;
    std::unordered_map<char, size_t> charToId_;

public:
    CharTokenizer();

    void buildFromText(const std::string& text);

    std::vector<size_t> encode(const std::string& text) const;
    std::string decode(const std::vector<size_t>& ids) const;

    size_t vocabSize() const;

    char idToChar(size_t id) const;
    size_t charToId(char c) const;
};