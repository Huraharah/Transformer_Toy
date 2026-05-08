#pragma once

#include "core/tensor.h"
#include "core/random.h"
#include "layers/layer_norm.h"
#include "layers/attention.h"
#include "layers/ffn.h"

class TransformerBlock {
private:
    size_t embedDim_;
    size_t hiddenDim_;

    LayerNorm norm1_;
    SelfAttention attention_;

    LayerNorm norm2_;
    FFN ffn_;

public:
    TransformerBlock(size_t embedDim, size_t hiddenDim, Random& rng);

    Tensor forward(const Tensor& input) const;
};