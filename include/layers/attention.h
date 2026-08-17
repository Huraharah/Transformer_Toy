#pragma once

#include "core/tensor.h"
#include "core/random.h"
#include "layers/linear.h"
#include "core/layer_utils.h"
#include "layers/config.h"
#include "layers/dropout.h"

class TrainingProfiler;

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
    AttentionConfig config_;

    size_t embedDim_;

    Linear queryProj_;
    Linear keyProj_;
    Linear valueProj_;
    Linear outputProj_;

	Tensor cachedInput_;
	Tensor cachedQ_;
	Tensor cachedK_;
	Tensor cachedV_;
	Tensor cachedAttentionWeights_;

    TrainingProfiler* profiler_ = nullptr;

    Dropout attentionDropout_;
    Dropout projectionDropout_;
    bool training_ = true;

public:
    SelfAttention(const AttentionConfig& config, Random& rng);

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& gradOutput);

    void setProfiler(TrainingProfiler* profiler);

	std::vector<Parameter*> parameters();  

    void train();
    void eval();
    bool isTraining() const;
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

	Tensor cachedInput_;
    Tensor cachedQ_;
    Tensor cachedK_;
    Tensor cachedV_;
    Tensor cachedAttentionWeights_;

    TrainingProfiler* profiler_ = nullptr;

    Dropout attentionDropout_;
    Dropout projectionDropout_;
    bool training_ = true;

public:
    explicit MultiHeadAttention(const AttentionConfig& config, Random& rng);

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& gradOutput);

    void setProfiler(TrainingProfiler* profiler);

    std::vector<Parameter*> parameters();

    void train();
    void eval();
    bool isTraining() const;
};
