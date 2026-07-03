#pragma once

#include "core/tensor.h"
#include "core/random.h"
#include "layers/linear.h"
#include "core/layer_utils.h"
#include "layers/config.h"


inline std::string attentionTypeToString(AttentionType type) {
    switch (type) {
    case AttentionType::SingleHead:
        return "SingleHead";

    case AttentionType::MultiHead:
        return "MultiHead";

    default:
        return "Unknown";
    }
}

class SelfAttention {
private:
    size_t embedDim_;

    Linear queryProj_;
    Linear keyProj_;
    Linear valueProj_;
    Linear outputProj_;

public:
    SelfAttention(size_t embedDim, Random& rng);

    Tensor forward(const Tensor& input) const;
	std::vector<Parameter*> parameters();   
};

class MultiHeadAttention {
private:
    AttentionConfig config_;

    size_t embedDim_;
    size_t numHeads_;
    size_t headDim_;

    Linear queryProj_;
    Linear keyProj_;
    Linear valueProj_;
    Linear outputProj_;

public:
    explicit MultiHeadAttention(const AttentionConfig& config, Random& rng);

    Tensor forward(const Tensor& input) const;
    std::vector<Parameter*> parameters();
};
