#include "layers/transformer_block.h"
#include "core/math_utils.h"

#include <stdexcept>

TransformerBlock::TransformerBlock(size_t embedDim, size_t hiddenDim, Random& rng)
    : embedDim_(embedDim),
    hiddenDim_(hiddenDim),
    norm1_(embedDim),
    attention_(embedDim, rng),
    norm2_(embedDim),
    ffn_(embedDim, hiddenDim, rng) {
}

Tensor TransformerBlock::forward(const Tensor& input) const {
    if (input.rank() != 3) {
        throw std::invalid_argument(
            "TransformerBlock::forward expects input shape [batch, sequence, embedDim]."
        );
    }

    if (input.shape()[2] != embedDim_) {
        throw std::invalid_argument(
            "TransformerBlock embed dimension mismatch."
        );
    }

    Tensor normed1 = norm1_.forward(input);
    Tensor attended = attention_.forward(normed1);
    Tensor residual1 = MathUtils::add(input, attended);

    Tensor normed2 = norm2_.forward(residual1);
    Tensor mixed = ffn_.forward(normed2);
    Tensor residual2 = MathUtils::add(residual1, mixed);

    return residual2;
}