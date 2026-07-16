#include "layers/transformer_block.h"
#include "core/math_utils.h"
#include "core/parameter.h"

#include <stdexcept>
#include <training/training_profiler.h>

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

void TransformerBlock::setProfiler(TrainingProfiler* profiler) {
    profiler_ = profiler;

    ffn_.setProfiler(profiler);
    norm1_.setProfiler(profiler);
    norm2_.setProfiler(profiler);

    if (singleAttention_) {
        singleAttention_->setProfiler(profiler);
    }

    if (multiAttention_) {
        multiAttention_->setProfiler(profiler);
    }
}

Tensor TransformerBlock::forward(
    const Tensor& input
) {
    if (input.rank() != 3) {
        throw std::invalid_argument(
            "TransformerBlock::forward expects input shape "
            "[batch, sequence, embedDim]."
        );
    }

    if (input.shape()[2] != embedDim_) {
        throw std::invalid_argument(
            "TransformerBlock embed dimension mismatch."
        );
    }

    cachedInput_ = input;

    Tensor normed1;
    Tensor attended;
    Tensor residual1;
    Tensor normed2;
    Tensor mixed;
    Tensor residual2;

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::BlockNorm1Forward,
            input.device()
        );

        normed1 =
            norm1_.forward(input);
    }

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::BlockAttentionForward,
            input.device()
        );

        if (singleAttention_) {
            attended =
                singleAttention_->forward(
                    normed1
                );
        }
        else if (multiAttention_) {
            attended =
                multiAttention_->forward(
                    normed1
                );
        }
        else {
            throw std::runtime_error(
                "TransformerBlock: no attention layer initialized."
            );
        }
    }

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::BlockResidual1Forward,
            input.device()
        );

        residual1 =
            MathUtils::add(
                input,
                attended
            );

        cachedAttentionResidual_ =
            residual1;
    }

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::BlockNorm2Forward,
            input.device()
        );

        normed2 =
            norm2_.forward(
                residual1
            );
    }

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::BlockFFNForward,
            input.device()
        );

        mixed =
            ffn_.forward(
                normed2
            );
    }

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::BlockResidual2Forward,
            input.device()
        );

        residual2 =
            MathUtils::add(
                residual1,
                mixed
            );
    }

    return residual2;
}

Tensor TransformerBlock::backward(
    const Tensor& gradOutput
) {
    if (cachedInput_.empty()) {
        throw std::runtime_error(
            "TransformerBlock::backward called before forward."
        );
    }

    if (
        gradOutput.shape() !=
        cachedInput_.shape()
        ) {
        throw std::invalid_argument(
            "TransformerBlock::backward gradOutput shape mismatch."
        );
    }

    Tensor gradResidual1Direct =
        gradOutput;

    Tensor gradMixed =
        gradOutput;

    Tensor gradNormed2;
    Tensor gradResidual1FromFFN;
    Tensor gradResidual1;
    Tensor gradInputDirect;
    Tensor gradAttended;
    Tensor gradNormed1;
    Tensor gradInputFromAttention;
    Tensor gradInput;

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::BlockFFNBackward,
            gradOutput.device()
        );

        gradNormed2 =
            ffn_.backward(
                gradMixed
            );
    }

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::BlockNorm2Backward,
            gradOutput.device()
        );

        gradResidual1FromFFN =
            norm2_.backward(
                gradNormed2
            );
    }

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::BlockResidual1Backward,
            gradOutput.device()
        );

        gradResidual1 =
            MathUtils::add(
                gradResidual1Direct,
                gradResidual1FromFFN
            );
    }

    gradInputDirect =
        gradResidual1;

    gradAttended =
        gradResidual1;

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::BlockAttentionBackward,
            gradOutput.device()
        );

        if (singleAttention_) {
            gradNormed1 =
                singleAttention_->backward(
                    gradAttended
                );
        }
        else if (multiAttention_) {
            gradNormed1 =
                multiAttention_->backward(
                    gradAttended
                );
        }
        else {
            throw std::runtime_error(
                "TransformerBlock::backward has no attention module."
            );
        }
    }

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::BlockNorm1Backward,
            gradOutput.device()
        );

        gradInputFromAttention =
            norm1_.backward(
                gradNormed1
            );
    }

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::BlockInputMergeBackward,
            gradOutput.device()
        );

        gradInput =
            MathUtils::add(
                gradInputDirect,
                gradInputFromAttention
            );
    }

    return gradInput;
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