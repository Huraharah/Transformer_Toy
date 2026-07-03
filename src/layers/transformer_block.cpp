#include "layers/transformer_block.h"
#include "core/math_utils.h"
#include "core/parameter.h"

#include <stdexcept>

TransformerBlock::TransformerBlock(const TransformerBlockConfig& config, Random& rng)
    : config_(config),
    embedDim_(config.d_model),
    hiddenDim_(config.d_ff),
    norm1_(config.d_model),
    norm2_(config.d_model),
    ffn_(config.d_model, config.d_ff, rng) {

    config_.validate();

    if (config_.attentionType == AttentionType::SingleHead) {
        singleAttention_ = std::make_unique<SelfAttention>(
            config_.d_model,
            rng
        );
    }
    else if (config_.attentionType == AttentionType::MultiHead) {
        AttentionConfig attnCfg(
            config_.d_model,
            config_.numHeads,
            true
        );

        multiAttention_ = std::make_unique<MultiHeadAttention>(
            attnCfg,
            rng
        );
    }
    else {
        throw std::invalid_argument("TransformerBlock: unknown AttentionType.");
    }
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

    Tensor attended;

    if (singleAttention_) {
        attended = singleAttention_->forward(normed1);
    }
    else if (multiAttention_) {
        attended = multiAttention_->forward(normed1);
    }
    else {
        throw std::runtime_error("TransformerBlock: no attention layer initialized.");
    }

    Tensor residual1 = MathUtils::add(input, attended);

    Tensor normed2 = norm2_.forward(residual1);
    Tensor mixed = ffn_.forward(normed2);
    Tensor residual2 = MathUtils::add(residual1, mixed);

    return residual2;
}

std::vector<Parameter*> TransformerBlock::parameters() {
    std::vector<Parameter*> params;

    auto append = [&params](std::vector<Parameter*> more) {
        params.insert(params.end(), more.begin(), more.end());
        };

    if (singleAttention_) {
        append(singleAttention_->parameters());
    }

    if (multiAttention_) {
        append(multiAttention_->parameters());
    }

    append(norm1_.parameters());
    append(ffn_.parameters());
    append(norm2_.parameters());

    return params;
}