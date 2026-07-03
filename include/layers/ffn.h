#pragma once

#include "core/tensor.h"
#include "core/random.h"
#include "layers/linear.h"

class FFN {
private:
    size_t embedDim_;
    size_t hiddenDim_;

    Linear linear1_; // embedDim -> hiddenDim
    Linear linear2_; // hiddenDim -> embedDim

    Tensor cachedHiddenPreActivation_;
    std::vector<size_t> cachedInputShape_;

public:
    FFN(size_t embedDim, size_t hiddenDim, Random& rng);

    Tensor forward(const Tensor& input);

    Tensor backward(const Tensor& gradOutput);

    std::vector<Parameter*> parameters();
};