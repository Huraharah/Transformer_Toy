#pragma once

#include "core/tensor.h"
#include "core/random.h"
#include "layers/layer_norm.h"
#include "layers/attention.h"
#include "layers/ffn.h"
#include "layers/config.h"
#include "core/parameter.h"


class TransformerBlock {
private:
    TransformerBlockConfig config_;

    size_t embedDim_;
    size_t hiddenDim_;

    LayerNorm norm1_;
    LayerNorm norm2_;
    FFN ffn_;

	Tensor cachedInput_;
    Tensor cachedAttentionResidual_;

    TrainingProfiler* profiler_ = nullptr;

    std::unique_ptr<SelfAttention> singleAttention_;
    std::unique_ptr<MultiHeadAttention> multiAttention_;

public:
    TransformerBlock(const TransformerBlockConfig& config, Random& rng);

    Tensor forward(const Tensor& input);
	Tensor backward(const Tensor& gradOutput);

    void setProfiler(TrainingProfiler* profiler);

    std::vector<Parameter*> parameters();
};