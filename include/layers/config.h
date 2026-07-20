#pragma once

#include "core/random.h"
#include <stdexcept>
#include <string>

enum class AttentionType {
    SingleHead,
    MultiHead
};

struct AttentionConfig {
    size_t embedDim = 0;
    size_t numHeads = 0;
    size_t headDim() const {
        validate();
        return embedDim / numHeads;
    }

    bool causal = false;
    bool use_bias = true;

    float attention_dropout = 0.0f;
    float projection_dropout = 0.0f;

    AttentionConfig() = default;

    AttentionConfig(
        size_t embedDim_,
        size_t numHeads_,
        bool causal_ = false,
        bool use_bias_ = true,
        float attention_dropout_ = 0.0f,
        float projection_dropout_ = 0.0f
    )
        : embedDim(embedDim_),
        numHeads(numHeads_),
        causal(causal_),
        use_bias(use_bias_),
        attention_dropout(attention_dropout_),
        projection_dropout(projection_dropout_) {
        validate();
        headDim();
    }

    void validate() const {
        if (embedDim <= 0) {
            throw std::invalid_argument("AttentionConfig: embedDim must be > 0.");
        }

        if (numHeads <= 0) {
            throw std::invalid_argument("AttentionConfig: numHeads must be > 0.");
        }

        if (embedDim % numHeads != 0) {
            throw std::invalid_argument(
                "AttentionConfig: embedDim must be divisible by numHeads. Got embedDim=" +
                std::to_string(embedDim) + ", numHeads=" + std::to_string(numHeads) + "."
            );
        }

        if (attention_dropout < 0.0f || attention_dropout >= 1.0f) {
            throw std::invalid_argument("AttentionConfig: attention_dropout must be in [0.0, 1.0).");
        }

        if (projection_dropout < 0.0f || projection_dropout >= 1.0f) {
            throw std::invalid_argument("AttentionConfig: projection_dropout must be in [0.0, 1.0).");
        }
    }
};

struct TransformerBlockConfig {
    int d_model = 0;
    int d_ff = 0;

    bool pre_norm = true;
    bool use_bias = true;

    float residual_dropout = 0.0f;
    float ffn_dropout = 0.0f;

	AttentionType attentionType = AttentionType::SingleHead;
    size_t numHeads = 1;

    TransformerBlockConfig() = default;

    TransformerBlockConfig(
        int d_model_,
        int d_ff_,
        float residual_dropout_ = 0.0f,
        float ffn_dropout_ = 0.0f,
        bool pre_norm_ = true,
        bool use_bias_ = true
    )
        : d_model(d_model_),
        d_ff(d_ff_),
        pre_norm(pre_norm_),
        use_bias(use_bias_),
        residual_dropout(residual_dropout_),
        ffn_dropout(ffn_dropout_) {
        validate();
    }

    void validate() const {
        if (d_model <= 0) {
            throw std::invalid_argument("TransformerBlockConfig: d_model must be > 0.");
        }

        if (d_ff <= 0) {
            throw std::invalid_argument("TransformerBlockConfig: d_ff must be > 0.");
        }

        if (residual_dropout < 0.0f || residual_dropout >= 1.0f) {
            throw std::invalid_argument("TransformerBlockConfig: residual_dropout must be in [0.0, 1.0).");
        }

        if (ffn_dropout < 0.0f || ffn_dropout >= 1.0f) {
            throw std::invalid_argument("TransformerBlockConfig: ffn_dropout must be in [0.0, 1.0).");
        }

        if (attentionType == AttentionType::SingleHead && numHeads != 1) {
            throw std::invalid_argument(
                "TransformerBlockConfig: SingleHead attention requires numHeads == 1."
            );
        }

        if (attentionType == AttentionType::MultiHead) {
            if (numHeads == 0) {
                throw std::invalid_argument(
                    "TransformerBlockConfig: MultiHead attention requires numHeads > 0."
                );
            }

            if (d_model % numHeads != 0) {
                throw std::invalid_argument(
                    "TransformerBlockConfig: d_model must be divisible by numHeads for MultiHead attention."
                );
            }
        }
    }
};

struct TransformerModelConfig {
    int vocab_size = 0;
    int max_seq_len = 0;
    int num_layers = 0;

    TransformerBlockConfig block;

    bool learned_positional_embeddings = true;
    float embedding_dropout = 0.0f;

    TransformerModelConfig() = default;

    TransformerModelConfig(
        int vocab_size_,
        int max_seq_len_,
        int num_layers_,
        const TransformerBlockConfig& block_,
        bool learned_positional_embeddings_ = true,
        float embedding_dropout_ = 0.0f
    )
        : vocab_size(vocab_size_),
        max_seq_len(max_seq_len_),
        num_layers(num_layers_),
        block(block_),
        learned_positional_embeddings(learned_positional_embeddings_),
        embedding_dropout(embedding_dropout_) {
        validate();
    }

    void validate() const {
        if (vocab_size <= 0) {
            throw std::invalid_argument("TransformerModelConfig: vocab_size must be > 0.");
        }

        if (max_seq_len <= 0) {
            throw std::invalid_argument("TransformerModelConfig: max_seq_len must be > 0.");
        }

        if (num_layers <= 0) {
            throw std::invalid_argument("TransformerModelConfig: num_layers must be > 0.");
        }

        if (embedding_dropout < 0.0f || embedding_dropout >= 1.0f) {
            throw std::invalid_argument("TransformerModelConfig: embedding_dropout must be in [0.0, 1.0).");
        }

        block.validate();
    }
};

struct GenerationConfig {
    size_t maxNewTokens = 32;

    float temperature = 1.0f;
    size_t topK = 0;

    bool greedySampling = false;

    bool printGeneratedText = true;
    bool printTokenIds = false;

    unsigned int randomSeed = 42;

    GenerationConfig() = default;

    GenerationConfig(
        size_t maxNewTokens_,
        float temperature_ = 1.0f,
        size_t topK_ = 0,
        bool greedySampling_ = false,
        bool printGeneratedText_ = true,
        bool printTokenIds_ = false,
        unsigned int randomSeed_ = 42
    )
        : maxNewTokens(maxNewTokens_),
        temperature(temperature_),
        topK(topK_),
        greedySampling(greedySampling_),
        printGeneratedText(printGeneratedText_),
        printTokenIds(printTokenIds_),
        randomSeed(randomSeed_) {

        validate();
    }

    void validate() const {
        if (maxNewTokens == 0) {
            throw std::invalid_argument(
                "GenerationConfig: maxNewTokens must be > 0."
            );
        }

        if (temperature <= 0.0f) {
            throw std::invalid_argument(
                "GenerationConfig: temperature must be > 0."
            );
        }

        if (topK > 0 && topK < 2) {
            throw std::invalid_argument(
                "GenerationConfig: topK must be 0 or >= 2."
            );
        }
    }
};