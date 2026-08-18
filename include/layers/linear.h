#pragma once

#include "core/tensor.h"
#include "core/random.h"
#include "core/parameter.h"

class Linear {
private:
    size_t inFeatures_;
    size_t outFeatures_;
    bool useBias_;

    Parameter weights_; // [outFeatures, inFeatures]
    Parameter bias_;    // [outFeatures]

	Tensor cachedInput_; // [batchSize, inFeatures]

public:
    Linear(size_t inFeatures, size_t outFeatures, Random& rng, bool useBias = true);

    Tensor forward(const Tensor& input);
    Tensor backward(Tensor& gradOutput);


    const Tensor& weights() const;
    const Tensor& bias() const;
    std::vector<Parameter*> parameters();
};