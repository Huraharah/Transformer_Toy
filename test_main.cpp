#include "core/tensor.h"
#include "core/random.h"
#include "layers/linear.h"
#include "layers/embedding.h"
#include "layers/layer_norm.h"
#include "core/math_utils.h"
#include "layers/attention.h"
#include "layers/ffn.h"
#include "layers/transformer_block.h"
#include "model/transformer.h"
#include "data/tokenizer.h"
#include "data/dataset.h"
#include "layers/config.h"
#include "training/sgd_optimizer.h"
#include "core/parameter.h"
#include "training/adam_optimizer.h"
#include "training/cross_entropy_loss.h"
#include "training/training_config.h"
#include "training/training_history.h"
#include "training/checkpoint.h"
#include "training/trainer.h"
#include "training/checkpoint_callback.h"
#include "training/best_checkpoint_callback.h"
#include "training/early_stopping_callback.h"
#include "training/generation_callback.h"
#include "tests/best_model_generation_tests.h"
#include <tests/profile_testing.h>
#include "kernels/tensor_ops_kernels.cuh"
#include "kernels/linear_kernels.cuh"
#include "kernels/activation_kernels.cuh"
#include "tests/smoke_test.h"
#include "tests/shakedown_test.h"

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <stdarg.h>

void coreTest() {
    Tensor t({ 2, 3 }, 1.0f);

    t.at({ 0, 0 }) = 5.0f;
    t.at({ 1, 2 }) = 9.0f;

    std::cout << "===================================================\n";
    std::cout << "||                 Tensor Test                   ||\n";
    std::cout << "===================================================\n";
    std::cout << "Initial tensor created with shape [2, 3] and filled with 1.0f.\n";
    std::cout << "Shape: " << t.shapeString() << "\n";
    std::cout << "Size: " << t.size() << "\n";
    std::cout << "t[0,0]: " << t.at({ 0, 0 }) << "\n";
    std::cout << "t[1,2]: " << t.at({ 1, 2 }) << "\n";

    t.reshape({ 3, 2 });
    std::cout << "New shape: " << t.shapeString() << "\n";
    std::cout << "Flat data: ";

    for (size_t i = 0; i < t.size(); ++i) {
        std::cout << t[i] << " ";
    }

    std::cout << "\n\n";

    std::cout << "==================================================\n";
    std::cout << "||                 Random Test                  ||\n";
    std::cout << "==================================================\n";

    Random rng(123);

    std::cout << "Uniform [-1, 1]: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << rng.uniform(-1.0f, 1.0f) << " ";
    }
    std::cout << "\n";

    rng.setSeed(123);

    std::cout << "Repeat uniform [-1, 1]: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << rng.uniform(-1.0f, 1.0f) << " ";
    }
    std::cout << "\n";

    std::cout << "Normal [0, 1]: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << rng.normal(0.0f, 1.0f) << " ";
    }
    std::cout << "\n";

    std::cout << "Random ints [0, 9]: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << rng.randint(0, 9) << " ";
    }
    std::cout << "\n\n";

    std::cout << "==================================================\n";
    std::cout << "||                 Linear Test                  ||\n";
    std::cout << "==================================================\n";

    Random linearRng(42);

    Linear linear(3, 2, linearRng);

    Tensor input({ 2, 3 }, 0.0f);

    input.at({ 0, 0 }) = 1.0f;
    input.at({ 0, 1 }) = 2.0f;
    input.at({ 0, 2 }) = 3.0f;

    input.at({ 1, 0 }) = 4.0f;
    input.at({ 1, 1 }) = 5.0f;
    input.at({ 1, 2 }) = 6.0f;

    Tensor output = linear.forward(input);

    std::cout << "Input shape: " << input.shapeString() << "\n";
    std::cout << "Weights shape: " << linear.weights().shapeString() << "\n";
    std::cout << "Bias shape: " << linear.bias().shapeString() << "\n";
    std::cout << "Output shape: " << output.shapeString() << "\n";

    std::cout << "Output values:\n";
    for (size_t b = 0; b < output.shape()[0]; ++b) {
        for (size_t o = 0; o < output.shape()[1]; ++o) {
            std::cout << output.at({ b, o }) << " ";
        }
        std::cout << "\n\n";
    }

    std::cout << "==================================================\n";
    std::cout << "||                Embedding Test                ||\n";
    std::cout << "==================================================\n";

    Random embeddingRng(7);

    Embedding embedding(5, 4, embeddingRng);
    // vocabSize = 5 tokens, embeddingDim = 4

    Tensor tokenIds({ 2, 3 }, 0.0f);
    // batch 0: 0, 1, 2
    tokenIds.at({ 0, 0 }) = 0;
    tokenIds.at({ 0, 1 }) = 1;
    tokenIds.at({ 0, 2 }) = 2;

    // batch 1: 2, 3, 4
    tokenIds.at({ 1, 0 }) = 2;
    tokenIds.at({ 1, 1 }) = 3;
    tokenIds.at({ 1, 2 }) = 4;

    Tensor embedded = embedding.forward(tokenIds);

    std::cout << "Token ID shape: " << tokenIds.shapeString() << "\n";
    std::cout << "Embedding table shape: " << embedding.table().shapeString() << "\n";
    std::cout << "Embedded output shape: " << embedded.shapeString() << "\n";

    std::cout << "Embedded vectors:\n";
    for (size_t b = 0; b < embedded.shape()[0]; ++b) {
        std::cout << "Batch " << b << ":\n";

        for (size_t t = 0; t < embedded.shape()[1]; ++t) {
            std::cout << "  token " << tokenIds.at({ b, t }) << ": ";

            for (size_t e = 0; e < embedded.shape()[2]; ++e) {
                std::cout << embedded.at({ b, t, e }) << " ";
            }

            std::cout << "\n\n";
        }
    }

    std::cout << "==================================================\n";
    std::cout << "||               LayerNorm Test                 ||\n";
    std::cout << "==================================================\n";

    Tensor normInput({ 1, 2, 4 }, 0.0f);

    // token 0
    normInput.at({ 0, 0, 0 }) = 2.0f;
    normInput.at({ 0, 0, 1 }) = 4.0f;
    normInput.at({ 0, 0, 2 }) = 6.0f;
    normInput.at({ 0, 0, 3 }) = 8.0f;

    // token 1
    normInput.at({ 0, 1, 0 }) = 1.0f;
    normInput.at({ 0, 1, 1 }) = 3.0f;
    normInput.at({ 0, 1, 2 }) = 5.0f;
    normInput.at({ 0, 1, 3 }) = 7.0f;

    LayerNorm layerNorm(4);

    Tensor normOutput = layerNorm.forward(normInput);

    std::cout << "Input shape: "
        << normInput.shapeString() << "\n";

    std::cout << "Output shape: "
        << normOutput.shapeString() << "\n";

    for (size_t t = 0; t < normOutput.shape()[1]; ++t) {
        std::cout << "Token " << t << ": ";

        for (size_t f = 0; f < normOutput.shape()[2]; ++f) {
            std::cout << normOutput.at({ 0, t, f }) << " ";
        }

        std::cout << "\n\n";
    }

    std::cout << "==================================================\n";
    std::cout << "||                MathUtils Test                ||\n";
    std::cout << "==================================================\n";

    std::vector<float> scores = { 2.0f, 1.0f, 0.1f };
    std::vector<float> probs = MathUtils::softmax(scores);

    float sum = 0.0f;

    std::cout << "Softmax: ";
    for (float p : probs) {
        std::cout << p << " ";
        sum += p;
    }
    std::cout << "\n";

    std::cout << "Softmax sum: " << sum << "\n";

    std::cout << "ReLU(-2): " << MathUtils::relu(-2.0f) << "\n";
    std::cout << "ReLU(3): " << MathUtils::relu(3.0f) << "\n";

    std::cout << "GELU(-1): " << MathUtils::gelu(-1.0f) << "\n";
    std::cout << "GELU(0): " << MathUtils::gelu(0.0f) << "\n";
    std::cout << "GELU(1): " << MathUtils::gelu(1.0f) << "\n\n";

}

void layersTest() {
    std::cout << "==================================================\n";
    std::cout << "||            Self-Attention Test               ||\n";
    std::cout << "==================================================\n";

    Random attentionRng(99);

    SelfAttention attention(4, attentionRng);

    Tensor attentionInput({ 1, 3, 4 }, 0.0f);

    attentionInput.at({ 0, 0, 0 }) = 1.0f;
    attentionInput.at({ 0, 0, 1 }) = 0.0f;
    attentionInput.at({ 0, 0, 2 }) = 0.0f;
    attentionInput.at({ 0, 0, 3 }) = 0.0f;

    attentionInput.at({ 0, 1, 0 }) = 0.0f;
    attentionInput.at({ 0, 1, 1 }) = 1.0f;
    attentionInput.at({ 0, 1, 2 }) = 0.0f;
    attentionInput.at({ 0, 1, 3 }) = 0.0f;

    attentionInput.at({ 0, 2, 0 }) = 0.0f;
    attentionInput.at({ 0, 2, 1 }) = 0.0f;
    attentionInput.at({ 0, 2, 2 }) = 1.0f;
    attentionInput.at({ 0, 2, 3 }) = 0.0f;

    Tensor attentionOutput = attention.forward(attentionInput);

    std::cout << "Input shape: " << attentionInput.shapeString() << "\n";
    std::cout << "Output shape: " << attentionOutput.shapeString() << "\n";

    std::cout << "Attention output:\n";
    for (size_t t = 0; t < attentionOutput.shape()[1]; ++t) {
        std::cout << "Token " << t << ": ";

        for (size_t f = 0; f < attentionOutput.shape()[2]; ++f) {
            std::cout << attentionOutput.at({ 0, t, f }) << " ";
        }

        std::cout << "\n\n";
    }

    std::cout << "==================================================\n";
    std::cout << "||                  FFN Test                    ||\n";
    std::cout << "==================================================\n";

    Random ffnRng(1234);

    FFN ffn(4, 16, ffnRng);
    // embedDim = 4, hiddenDim = 16

    Tensor ffnInput({ 1, 3, 4 }, 0.0f);

    ffnInput.at({ 0, 0, 0 }) = 1.0f;
    ffnInput.at({ 0, 0, 1 }) = 0.0f;
    ffnInput.at({ 0, 0, 2 }) = 0.0f;
    ffnInput.at({ 0, 0, 3 }) = 0.0f;

    ffnInput.at({ 0, 1, 0 }) = 0.0f;
    ffnInput.at({ 0, 1, 1 }) = 1.0f;
    ffnInput.at({ 0, 1, 2 }) = 0.0f;
    ffnInput.at({ 0, 1, 3 }) = 0.0f;

    ffnInput.at({ 0, 2, 0 }) = 0.0f;
    ffnInput.at({ 0, 2, 1 }) = 0.0f;
    ffnInput.at({ 0, 2, 2 }) = 1.0f;
    ffnInput.at({ 0, 2, 3 }) = 0.0f;

    Tensor ffnOutput = ffn.forward(ffnInput);

    std::cout << "Input shape: " << ffnInput.shapeString() << "\n";
    std::cout << "Output shape: " << ffnOutput.shapeString() << "\n";

    std::cout << "FFN output:\n";
    for (size_t t = 0; t < ffnOutput.shape()[1]; ++t) {
        std::cout << "Token " << t << ": ";

        for (size_t f = 0; f < ffnOutput.shape()[2]; ++f) {
            std::cout << ffnOutput.at({ 0, t, f }) << " ";
        }

        std::cout << "\n\n";
    }

    std::cout << "==================================================\n";
    std::cout << "||             TransformerBlock Test            ||\n";
    std::cout << "==================================================\n";

    Random blockRng(2024);

    TransformerBlockConfig blockConfig(4, 16, 0.1f, 0.1f, true, true);

    TransformerBlock block(blockConfig, blockRng);

    Tensor blockInput({ 1, 3, 4 }, 0.0f);

    blockInput.at({ 0, 0, 0 }) = 1.0f;
    blockInput.at({ 0, 0, 1 }) = 0.0f;
    blockInput.at({ 0, 0, 2 }) = 0.0f;
    blockInput.at({ 0, 0, 3 }) = 0.0f;

    blockInput.at({ 0, 1, 0 }) = 0.0f;
    blockInput.at({ 0, 1, 1 }) = 1.0f;
    blockInput.at({ 0, 1, 2 }) = 0.0f;
    blockInput.at({ 0, 1, 3 }) = 0.0f;

    blockInput.at({ 0, 2, 0 }) = 0.0f;
    blockInput.at({ 0, 2, 1 }) = 0.0f;
    blockInput.at({ 0, 2, 2 }) = 1.0f;
    blockInput.at({ 0, 2, 3 }) = 0.0f;

    Tensor blockOutput = block.forward(blockInput);

    std::cout << "Input shape: " << blockInput.shapeString() << "\n";
    std::cout << "Output shape: " << blockOutput.shapeString() << "\n";

    std::cout << "TransformerBlock output:\n";
    for (size_t t = 0; t < blockOutput.shape()[1]; ++t) {
        std::cout << "Token " << t << ": ";

        for (size_t f = 0; f < blockOutput.shape()[2]; ++f) {
            std::cout << blockOutput.at({ 0, t, f }) << " ";
        }

        std::cout << "\n\n";
    }

    std::cout << "==================================================\n";
    std::cout << "||              Transformer Model Test          ||\n";
    std::cout << "==================================================\n";

    Random modelRng(777);
    TransformerModelConfig modelConfig(10, 4, 2, blockConfig, true, 0.0f);

    Transformer model(
        modelConfig,
        modelRng
    );

    try {
        Tensor modelInput({ 1, 4 }, 0.0f);

        modelInput.at({ 0, 0 }) = 1;
        modelInput.at({ 0, 1 }) = 2;
        modelInput.at({ 0, 2 }) = 3;
        modelInput.at({ 0, 3 }) = 4;

        Tensor logits = model.forward(modelInput);

        std::cout << "Input shape: " << modelInput.shapeString() << "\n";
        std::cout << "Logits shape: " << logits.shapeString() << "\n";

        for (size_t t = 0; t < logits.shape()[1]; ++t) {
            std::cout << "Token position " << t << " logits: ";

            for (size_t v = 0; v < logits.shape()[2]; ++v) {
                std::cout << logits.at({ 0, t, v }) << " ";
            }

            std::cout << "\n\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error during Transformer forward pass: " << e.what() << "\n";
    }

    std::cout << "==================================================\n";
    std::cout << "||               Tokenizer Test                 ||\n";
    std::cout << "==================================================\n";

    std::string text = "hello transformer";

    CharTokenizer tokenizer;
    tokenizer.buildFromText(text);

    std::vector<int> encoded = tokenizer.encode(text);
    std::string decoded = tokenizer.decode(encoded);

    std::cout << "Original text: " << text << "\n";
    std::cout << "Vocab size: " << tokenizer.vocabSize() << "\n";

    std::cout << "Encoded IDs: ";
    for (size_t id : encoded) {
        std::cout << id << " ";
    }
    std::cout << "\n";

    std::cout << "Decoded text: " << decoded << "\n\n";
}
void dataTest() {

    std::cout << "==================================================\n";
    std::cout << "||                Dataset Test                  ||\n";
    std::cout << "==================================================\n";

    TextDataset dataset("data/shakespeare.txt", 16);

    std::cout << "Raw text length: " << dataset.rawText().size() << "\n";
    std::cout << "Vocab size: " << dataset.vocabSize() << "\n";
    std::cout << "Context length: " << dataset.contextLength() << "\n";
    std::cout << "Num windows: " << dataset.numWindows() << "\n";

    Tensor sampleInput = dataset.getInputWindow(0);
    Tensor sampleTarget = dataset.getTargetWindow(0);

    std::vector<size_t> inputIds;
    std::vector<size_t> targetIds;

    for (size_t i = 0; i < dataset.contextLength(); ++i) {
        inputIds.push_back(static_cast<size_t>(sampleInput.at({ 0, i })));
        targetIds.push_back(static_cast<size_t>(sampleTarget.at({ 0, i })));
    }

    //std::cout << "Input text:  [" << dataset.tokenizer().decode(inputIds) << "]\n";
    //std::cout << "Target text: [" << dataset.tokenizer().decode(targetIds) << "]\n\n";

    std::cout << "==================================================\n";
    std::cout << "||              Generation Test                 ||\n";
    std::cout << "==================================================\n";

    /*Random genRng(987);

    Transformer genModel(
        dataset.vocabSize(),
        32,
        32,
        128,
        2,
        genRng
    );

    std::string prompt = "First Citizen:";

    GenerationConfig genConfig(
        80,     // maxNewTokens
        1.0f,   // temperature
        10,     // topK
        false,  // greedySampling
        true,   // printGeneratedText
        false,  // printTokenIds
        555     // randomSeed
    );

    Random sampleRng(genConfig.randomSeed);

    std::string generated = genModel.generate(
        prompt,
        dataset.tokenizer(),
        genConfig,
        sampleRng
    );

    std::cout << "Prompt:\n" << prompt << "\n\n";
    std::cout << "Generated:\n" << generated << "\n\n";*/
    std::cout << "Generation test is commented out to save time during testing. Uncomment to run.\n\n";

    std::cout << "==================================================\n";
    std::cout << "||                 Batch Test                   ||\n";
    std::cout << "==================================================\n";

    Random batchRng(2468);

    Tensor batchInputs;
    Tensor batchTargets;

    dataset.getBatch(4, batchRng, batchInputs, batchTargets);

    std::cout << "Batch input shape: " << batchInputs.shapeString() << "\n";
    std::cout << "Batch target shape: " << batchTargets.shapeString() << "\n";

    for (size_t b = 0; b < batchInputs.shape()[0]; ++b) {
        std::vector<size_t> inputIds;
        std::vector<size_t> targetIds;

        for (size_t t = 0; t < batchInputs.shape()[1]; ++t) {
            inputIds.push_back(static_cast<size_t>(batchInputs.at({ b, t })));
            targetIds.push_back(static_cast<size_t>(batchTargets.at({ b, t })));
        }

        //std::cout << "Sample " << b << " input:  ["
            //<< dataset.tokenizer().decode(inputIds) << "]\n";

        //std::cout << "Sample " << b << " target: ["
            //<< dataset.tokenizer().decode(targetIds) << "]\n";
    }

    /*std::cout << "\n==================================================\n";
    std::cout << "||                  Loss Test                   ||\n";
    std::cout << "==================================================\n";

    Random lossModelRng(1357);

    Transformer lossModel(
        dataset.vocabSize(),
        dataset.contextLength(),
        32,
        128,
        2,
        lossModelRng
    );

    Tensor lossInputs;
    const Tensor lossTargets;

    Random lossBatchRng(2468);
    dataset.getBatch(4, lossBatchRng, lossInputs, lossTargets);

    Tensor lossLogits = lossModel.forward(lossInputs);

    CrossEntropyLoss celoss(lossLogits, lossTargets);

    float loss = celoss.forward(lossLogits, lossTargets);
    float ppl = MathUtils::perplexity(loss);
    float acc = MathUtils::tokenAccuracy(lossLogits, lossTargets);

    std::cout << "Input shape: " << lossInputs.shapeString() << "\n";
    std::cout << "Target shape: " << lossTargets.shapeString() << "\n";
    std::cout << "Logits shape: " << lossLogits.shapeString() << "\n";
    std::cout << "Cross-entropy loss: " << loss << "\n";
    std::cout << "Perplexity: " << ppl << "\n";
    std::cout << "Token accuracy: " << acc << "\n";*/

    std::cout << "\n==================================================\n";
    std::cout << "||            All core tests completed!         ||\n";
    std::cout << "==================================================\n";
}

void configTest() {
    std::cout << "\n==================================================\n";
    std::cout << "||              Config Tests                    ||\n";
    std::cout << "==================================================\n";

    AttentionConfig cfg(512, 8, true);
    try {
        std::cout << "AttentionConfig test:\n";
        std::cout << "d_model: " << cfg.embedDim << "\n";
        std::cout << "num_heads: " << cfg.numHeads << "\n";
        std::cout << "d_head: " << cfg.headDim << "\n";
        std::cout << "causal: " << cfg.causal << "\n";
        assert(cfg.embedDim == 512);
        assert(cfg.numHeads == 8);
        assert(cfg.headDim == 64);
        assert(cfg.causal == true);
    }
    catch (const std::exception& e) {
        std::cerr << "AttentionConfig test failed: " << e.what() << "\n";
    }
    std::cout << "\n";

    TransformerBlockConfig blockCfg(512, 2048, 0.1f, 0.1f);
    try {
        std::cout << "TransformerBlockConfig test:\n";
        std::cout << "d_model: " << blockCfg.d_model << "\n";
        std::cout << "d_ff: " << blockCfg.d_ff << "\n";
        std::cout << "residual_dropout: " << blockCfg.residual_dropout << "\n";
        std::cout << "ffn_dropout: " << blockCfg.ffn_dropout << "\n";
        assert(blockCfg.d_model == 512);
        assert(blockCfg.d_ff == 2048);
        assert(blockCfg.residual_dropout == 0.1f);
        assert(blockCfg.ffn_dropout == 0.1f);
    }
    catch (const std::exception& e) {
        std::cerr << "TransformerBlockConfig test failed: " << e.what() << "\n";
    }
    std::cout << "\n";

    TransformerModelConfig modelCfg(64, 512, 8, blockCfg, true, 0.1f);
    try {
        std::cout << "TransformerModelConfig test:\n";
        std::cout << "vocab_size: " << modelCfg.vocab_size << "\n";
        std::cout << "max_seq_len: " << modelCfg.max_seq_len << "\n";
        std::cout << "num_layers: " << modelCfg.num_layers << "\n";
        std::cout << "block_config.learned_pos_emb: " << modelCfg.learned_positional_embeddings << "\n";
        std::cout << "block_config.embedding_dropout: " << modelCfg.embedding_dropout << "\n";
        assert(modelCfg.vocab_size == 64);
        assert(modelCfg.max_seq_len == 512);
        assert(modelCfg.num_layers == 8);
        assert(modelCfg.learned_positional_embeddings);
        assert(modelCfg.embedding_dropout == 0.1f);
    }
    catch (const std::exception& e) {
        std::cerr << "TransformerModelConfig test failed: " << e.what() << "\n";
    }

    std::cout << "\nTesting bad configs...\n\nAttention Config:\n";
    try {
        AttentionConfig bad1(512, 7);  // d_model not divisible by num_heads
        assert(false && "Expected exception for bad AttentionConfig 1 was not thrown.");
    }
    catch (const std::exception& e) {
        std::cerr << "Caught expected exception for bad AttentionConfig 1: " << e.what() << "\n";
    }
    try {
        AttentionConfig bad2(0, 8);    // d_model <= 0
        assert(false && "Expected exception for bad AttentionConfig 2 was not thrown.");
    }
    catch (const std::exception& e) {
        std::cerr << "Caught expected exception for bad AttentionConfig 2: " << e.what() << "\n";
    }
    try {
        AttentionConfig bad3(512, 8, true, true, -1.0f, 0.0f);
        assert(false && "Expected exception for bad AttentionConfig 3 was not thrown.");
    }
    catch (const std::exception& e) {
        std::cerr << "Caught expected exception for bad AttentionConfig 3: " << e.what() << "\n";
    }

    try {
        AttentionConfig bad4(512, 8, true, true, 0.0f, 1.5f);
        assert(false && "Expected exception for bad AttentionConfig 4 was not thrown.");
    }
    catch (const std::exception& e) {
        std::cerr << "Caught expected exception for bad AttentionConfig 4: " << e.what() << "\n";
    }
    try {
        AttentionConfig bad5(512, 0);    // num_heads <= 0
        assert(false && "Expected exception for bad AttentionConfig 5 was not thrown.");
    }
    catch (const std::exception& e) {
        std::cerr << "Caught expected exception for bad AttentionConfig 5: " << e.what() << "\n";
    }
    std::cout << "\nTransformerBlock Config:\n";
    try {
        TransformerBlockConfig bad1(512, 0);  // d_ff <= 0
        assert(false && "Expected exception for bad TransformerBlockConfig 1 was not thrown.");
    }
    catch (const std::exception& e) {
        std::cerr << "Caught expected exception for bad TransformerBlockConfig 1: " << e.what() << "\n";
    }
    try {
        TransformerBlockConfig bad2(0, 2048);  // d_model <= 0
        assert(false && "Expected exception for bad TransformerBlockConfig 2 was not thrown.");
    }
    catch (const std::exception& e) {
        std::cerr << "Caught expected exception for bad TransformerBlockConfig 2: " << e.what() << "\n";
    }
    try {
        TransformerBlockConfig bad3(512, 2048, -0.1f);  // negative residual dropout
        assert(false && "Expected exception for bad TransformerBlockConfig 3 was not thrown.");
    }
    catch (const std::exception& e) {
        std::cerr << "Caught expected exception for bad TransformerBlockConfig 3: " << e.what() << "\n\n";
    }

    std::cout << "GenerationConfig test:\n";

    GenerationConfig genCfg(64, 0.8f, 10, false, true, false, 123);

    assert(genCfg.maxNewTokens == 64);
    assert(genCfg.temperature == 0.8f);
    assert(genCfg.topK == 10);
    assert(genCfg.greedySampling == false);
    assert(genCfg.printGeneratedText == true);
    assert(genCfg.printTokenIds == false);
    assert(genCfg.randomSeed == 123);

    try {
        GenerationConfig badGenCfg(0);
        assert(false && "Expected exception for bad GenerationConfig maxNewTokens was not thrown.");
    }
    catch (const std::exception& e) {
        std::cout << "Caught expected GenerationConfig exception: " << e.what() << "\n";
    }

    try {
        GenerationConfig badTempCfg(32, 0.0f);
        assert(false && "Expected exception for bad GenerationConfig temperature was not thrown.");
    }
    catch (const std::exception& e) {
        std::cout << "Caught expected GenerationConfig exception: " << e.what() << "\n";
    }

    try {
        GenerationConfig badTopKCfg(32, 1.0f, 1);
        assert(false && "Expected exception for bad GenerationConfig topK was not thrown.");
    }
    catch (const std::exception& e) {
        std::cout << "Caught expected GenerationConfig exception: " << e.what() << "\n";
    }

    std::cout << "\nAll config tests completed.\n";
}

void mhaTest() {

    std::cout << "==================================================\n";
    std::cout << "||          Multi-Head Attention Test           ||\n";
    std::cout << "==================================================\n";

    Random mhaRng(99);

    AttentionConfig mhaConfig(
        4,      // embedDim
        2,      // numHeads
        true    // causal
    );

    MultiHeadAttention mha(mhaConfig, mhaRng);

    Tensor mhaInput({ 1, 3, 4 }, 0.0f);

    mhaInput.at({ 0, 0, 0 }) = 1.0f;
    mhaInput.at({ 0, 0, 1 }) = 0.0f;
    mhaInput.at({ 0, 0, 2 }) = 0.0f;
    mhaInput.at({ 0, 0, 3 }) = 0.0f;

    mhaInput.at({ 0, 1, 0 }) = 0.0f;
    mhaInput.at({ 0, 1, 1 }) = 1.0f;
    mhaInput.at({ 0, 1, 2 }) = 0.0f;
    mhaInput.at({ 0, 1, 3 }) = 0.0f;

    mhaInput.at({ 0, 2, 0 }) = 0.0f;
    mhaInput.at({ 0, 2, 1 }) = 0.0f;
    mhaInput.at({ 0, 2, 2 }) = 1.0f;
    mhaInput.at({ 0, 2, 3 }) = 0.0f;

    Tensor mhaOutput = mha.forward(mhaInput);

    std::cout << "Input shape: " << mhaInput.shapeString() << "\n";
    std::cout << "Output shape: " << mhaOutput.shapeString() << "\n";

    assert(mhaOutput.rank() == 3);
    assert(mhaOutput.shape()[0] == 1);
    assert(mhaOutput.shape()[1] == 3);
    assert(mhaOutput.shape()[2] == 4);

    std::cout << "Multi-head attention output:\n";
    for (size_t t = 0; t < mhaOutput.shape()[1]; ++t) {
        std::cout << "Token " << t << ": ";

        for (size_t f = 0; f < mhaOutput.shape()[2]; ++f) {
            std::cout << mhaOutput.at({ 0, t, f }) << " ";
        }

        std::cout << "\n\n";
    }

    try {
        AttentionConfig badMhaConfig(4, 3, true);
        MultiHeadAttention badMha(badMhaConfig, mhaRng);

        assert(false && "Expected exception for invalid multi-head config was not thrown.");
    }
    catch (const std::exception& e) {
        std::cout << "Caught expected MultiHeadAttention config exception: " << e.what() << "\n";
    }

    std::cout << "==================================================\n";
    std::cout << "||        TransformerBlock Multi-Head Test       ||\n";
    std::cout << "==================================================\n";

    Random mhaBlockRng(2024);

    TransformerBlockConfig mhaBlockConfig(4, 16, 0.1f, 0.1f, true, true);
    mhaBlockConfig.attentionType = AttentionType::MultiHead;
    mhaBlockConfig.numHeads = 2;
    mhaBlockConfig.validate();

    TransformerBlock mhaBlock(mhaBlockConfig, mhaBlockRng);

    Tensor mhaBlockInput({ 1, 3, 4 }, 0.0f);

    mhaBlockInput.at({ 0, 0, 0 }) = 1.0f;
    mhaBlockInput.at({ 0, 0, 1 }) = 0.0f;
    mhaBlockInput.at({ 0, 0, 2 }) = 0.0f;
    mhaBlockInput.at({ 0, 0, 3 }) = 0.0f;

    mhaBlockInput.at({ 0, 1, 0 }) = 0.0f;
    mhaBlockInput.at({ 0, 1, 1 }) = 1.0f;
    mhaBlockInput.at({ 0, 1, 2 }) = 0.0f;
    mhaBlockInput.at({ 0, 1, 3 }) = 0.0f;

    mhaBlockInput.at({ 0, 2, 0 }) = 0.0f;
    mhaBlockInput.at({ 0, 2, 1 }) = 0.0f;
    mhaBlockInput.at({ 0, 2, 2 }) = 1.0f;
    mhaBlockInput.at({ 0, 2, 3 }) = 0.0f;

    Tensor mhaBlockOutput = mhaBlock.forward(mhaBlockInput);

    std::cout << "Input shape: " << mhaBlockInput.shapeString() << "\n";
    std::cout << "Output shape: " << mhaBlockOutput.shapeString() << "\n";

    assert(mhaBlockOutput.rank() == 3);
    assert(mhaBlockOutput.shape()[0] == 1);
    assert(mhaBlockOutput.shape()[1] == 3);
    assert(mhaBlockOutput.shape()[2] == 4);

    std::cout << "TransformerBlock Multi-Head output:\n";
    for (size_t t = 0; t < mhaBlockOutput.shape()[1]; ++t) {
        std::cout << "Token " << t << ": ";

        for (size_t f = 0; f < mhaBlockOutput.shape()[2]; ++f) {
            std::cout << mhaBlockOutput.at({ 0, t, f }) << " ";
        }

        std::cout << "\n\n";
    }

    try {
        TransformerBlockConfig badMhaBlockConfig(4, 16, 0.1f, 0.1f, true, true);
        badMhaBlockConfig.attentionType = AttentionType::MultiHead;
        badMhaBlockConfig.numHeads = 3;
        badMhaBlockConfig.validate();

        TransformerBlock badMhaBlock(badMhaBlockConfig, mhaBlockRng);

        assert(false && "Expected exception for invalid TransformerBlock MHA config was not thrown.");
    }
    catch (const std::exception& e) {
        std::cout << "Caught expected TransformerBlock MHA config exception: "
            << e.what() << "\n\n";
    }

}

void testParameterBasics() {

	std::cout << "==================================================\n";
	std::cout << "||             Training Test Bench              ||\n";
	std::cout << "==================================================\n";

    Tensor w({ 2, 3 }, 1.0f);

    Parameter p(w, "test_weight");

    assert(p.name == "test_weight");
    assert(p.requires_grad);
    assert(p.value.size() == 6);
    assert(p.grad.size() == 6);

    for (size_t i = 0; i < p.grad.size(); ++i) {
        assert(p.grad[i] == 0.0f);
    }

    p.grad[0] = 5.0f;
    p.zeroGrad();

    for (size_t i = 0; i < p.grad.size(); ++i) {
        assert(p.grad[i] == 0.0f);
    }

    p.validate();

    std::cout << "\n[PASS] Parameter basics\n";
}

static bool near(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

void testOptimizerZeroGrad() {
    Tensor w({ 3 }, 1.0f);
    Parameter p(w, "w");

    p.grad[0] = 1.0f;
    p.grad[1] = -2.0f;
    p.grad[2] = 3.5f;

    std::vector<Parameter*> params = { &p };

    SGDOptimizer optimizer(0.1f);
    optimizer.zeroGrad(params);

    assert(near(p.grad[0], 0.0f));
    assert(near(p.grad[1], 0.0f));
    assert(near(p.grad[2], 0.0f));

    std::cout << "[PASS] Optimizer zeroGrad\n";
}

void testSGDOptimizerBasicStep() {
    Tensor w({ 3 }, 1.0f);
    Parameter p(w, "w");

    p.grad[0] = 0.25f;
    p.grad[1] = 0.50f;
    p.grad[2] = -1.00f;

    std::vector<Parameter*> params = { &p };

    SGDOptimizer optimizer(0.1f);
    optimizer.step(params);

    assert(near(p.value[0], 0.975f));
    assert(near(p.value[1], 0.950f));
    assert(near(p.value[2], 1.100f));

    std::cout << "[PASS] SGD optimizer basic step\n";
}

void testSGDOptimizerWeightDecay() {
    Tensor w({ 3 });
    w[0] = 1.0f;
    w[1] = 2.0f;
    w[2] = -3.0f;

    Parameter p(w, "w");

    p.grad[0] = 0.25f;
    p.grad[1] = 0.50f;
    p.grad[2] = -1.00f;

    std::vector<Parameter*> params = { &p };

    float lr = 0.1f;
    float weightDecay = 0.01f;

    SGDOptimizer optimizer(lr, weightDecay);
    optimizer.step(params);

    // effective_grad = grad + weightDecay * value
    assert(near(p.value[0], 1.0f - lr * (0.25f + 0.01f * 1.0f)));
    assert(near(p.value[1], 2.0f - lr * (0.50f + 0.01f * 2.0f)));
    assert(near(p.value[2], -3.0f - lr * (-1.00f + 0.01f * -3.0f)));

    std::cout << "[PASS] SGD optimizer weight decay\n";
}

void testAdamOptimizerFirstStep() {
    Tensor w({ 3 }, 1.0f);
    Parameter p(w, "w");

    p.grad[0] = 0.5f;
    p.grad[1] = -0.5f;
    p.grad[2] = 2.0f;

    std::vector<Parameter*> params = { &p };

    AdamOptimizer optimizer(
        0.001f,  // lr
        0.9f,    // beta1
        0.999f,  // beta2
        1e-8f,   // epsilon
        0.0f     // weight decay
    );

    optimizer.step(params);

    // On first Adam step with bias correction:
    // m_hat ≈ grad, v_hat ≈ grad^2
    // update ≈ lr * sign(grad)
    assert(near(p.value[0], 0.999f));
    assert(near(p.value[1], 1.001f));
    assert(near(p.value[2], 0.999f));

    std::cout << "[PASS] Adam optimizer first step\n";
}

void testAdamOptimizerMultipleStepsConstantGrad() {
    Tensor w({ 1 }, 1.0f);
    Parameter p(w, "w");

    std::vector<Parameter*> params = { &p };

    AdamOptimizer optimizer(0.001f);

    for (int step = 0; step < 5; ++step) {
        p.grad[0] = 0.5f;
        optimizer.step(params);
    }

    // With constant positive grad, Adam should reduce by ~lr each step.
    assert(near(p.value[0], 0.995f, 1e-5f));

    std::cout << "[PASS] Adam optimizer multiple constant-gradient steps\n";
}

void testAdamOptimizerWeightDecay() {
    Tensor w({ 1 }, 1.0f);
    Parameter p(w, "w");

    p.grad[0] = 0.5f;

    std::vector<Parameter*> params = { &p };

    AdamOptimizer optimizer(
        0.001f,
        0.9f,
        0.999f,
        1e-8f,
        0.1f   // weight decay
    );

    optimizer.step(params);

    // effective grad = 0.5 + 0.1 * 1.0 = 0.6
    // first Adam step still approximately lr in positive direction
    assert(near(p.value[0], 0.999f));

    std::cout << "[PASS] Adam optimizer weight decay\n";
}

void testAdamOptimizerZeroGradInherited() {
    Tensor w({ 2 }, 1.0f);
    Parameter p(w, "w");

    p.grad[0] = 3.0f;
    p.grad[1] = -4.0f;

    std::vector<Parameter*> params = { &p };

    AdamOptimizer optimizer(0.001f);
    optimizer.zeroGrad(params);

    assert(near(p.grad[0], 0.0f));
    assert(near(p.grad[1], 0.0f));

    std::cout << "[PASS] Adam optimizer inherited zeroGrad\n";
}

void testCrossEntropyLossPerfectConfidence() {
    Tensor logits({ 1, 3 });
    logits[0] = 10.0f;
    logits[1] = 0.0f;
    logits[2] = 0.0f;

    Tensor targets({ 1 });
    targets[0] = 0.0f;

    CrossEntropyLoss loss;

    float value = loss.forward(logits, targets);
    Tensor grad = loss.backward();

    assert(value < 0.001f);
    assert(grad.size() == logits.size());

    std::cout << "[PASS] CrossEntropyLoss perfect confidence\n";
}

void testCrossEntropyLossUniformLogits() {
    Tensor logits({ 1, 4 }, 0.0f);

    Tensor targets({ 1 });
    targets[0] = 2.0f;

    CrossEntropyLoss loss;

    float value = loss.forward(logits, targets);
    Tensor grad = loss.backward();

    assert(near(value, std::log(4.0f), 1e-5f));

    assert(near(grad[0], 0.25f));
    assert(near(grad[1], 0.25f));
    assert(near(grad[2], -0.75f));
    assert(near(grad[3], 0.25f));

    std::cout << "[PASS] CrossEntropyLoss uniform logits\n";
}

void testCrossEntropyLossBatchAverage() {
    Tensor logits({ 2, 3 });

    // sample 0: uniform logits, target 0
    logits[0] = 0.0f;
    logits[1] = 0.0f;
    logits[2] = 0.0f;

    // sample 1: confident target 2
    logits[3] = 0.0f;
    logits[4] = 0.0f;
    logits[5] = 10.0f;

    Tensor targets({ 2 });
    targets[0] = 0.0f;
    targets[1] = 2.0f;

    CrossEntropyLoss loss;

    float value = loss.forward(logits, targets);
    Tensor grad = loss.backward();

    float expected = (std::log(3.0f) + 0.0000908f) / 2.0f;

    assert(near(value, expected, 1e-4f));
    assert(grad.size() == logits.size());

    std::cout << "[PASS] CrossEntropyLoss batch average\n";
}

void testTrainingHistory() {
    TrainingHistory history;

    history.addTrainLoss(1.5f);
    history.addTrainLoss(1.2f);
    history.addValidationLoss(1.4f);

    assert(near(history.latestTrainLoss(), 1.2f));
    assert(near(history.latestValidationLoss(), 1.4f));

    history.saveCsv("training_history_test.csv");

    history.clear();

    assert(history.trainLosses.empty());
    assert(history.validationLosses.empty());

    std::cout << "[PASS] TrainingHistory basics\n";
}

void testCheckpointSaveLoad() {
    Tensor w1({ 2 });
    w1[0] = 1.5f;
    w1[1] = -2.0f;

    Tensor w2({ 2 });
    w2[0] = 3.25f;
    w2[1] = 4.75f;

    Parameter p1(w1, "layer.weight");
    Parameter p2(w2, "layer.bias");

    p1.grad[0] = 0.1f;
    p1.grad[1] = 0.2f;
    p2.grad[0] = -0.3f;
    p2.grad[1] = -0.4f;

    std::vector<Parameter*> params = { &p1, &p2 };

    TrainingHistory history;
    history.addTrainLoss(2.5f);
    history.addTrainLoss(1.75f);
    history.addValidationLoss(2.25f);

    CheckpointMetadata metadata;
    metadata.epoch = 3;
    metadata.globalStep = 42;
    metadata.runName = "checkpoint_test";

    Checkpoint::save(
        "checkpoint_test.bin",
        params,
        metadata,
        history
    );

    p1.value[0] = 0.0f;
    p1.value[1] = 0.0f;
    p2.value[0] = 0.0f;
    p2.value[1] = 0.0f;

    p1.grad[0] = 0.0f;
    p1.grad[1] = 0.0f;
    p2.grad[0] = 0.0f;
    p2.grad[1] = 0.0f;

    history.clear();
    metadata = CheckpointMetadata();

    Checkpoint::load(
        "checkpoint_test.bin",
        params,
        metadata,
        history
    );

    assert(near(p1.value[0], 1.5f));
    assert(near(p1.value[1], -2.0f));
    assert(near(p2.value[0], 3.25f));
    assert(near(p2.value[1], 4.75f));

    assert(near(p1.grad[0], 0.1f));
    assert(near(p1.grad[1], 0.2f));
    assert(near(p2.grad[0], -0.3f));
    assert(near(p2.grad[1], -0.4f));

    assert(metadata.epoch == 3);
    assert(metadata.globalStep == 42);
    assert(metadata.runName == "checkpoint_test");

    assert(history.trainLosses.size() == 2);
    assert(history.validationLosses.size() == 1);

    assert(near(history.trainLosses[0], 2.5f));
    assert(near(history.trainLosses[1], 1.75f));
    assert(near(history.validationLosses[0], 2.25f));

    std::cout << "[PASS] Checkpoint save/load\n";
}

class DummyTrainableModel : public TrainableModel {
private:
    Parameter logitsParam;

public:
    DummyTrainableModel()
        : logitsParam(Tensor({ 1, 3 }, 0.0f), "dummy.logits") {
    }

    Tensor forward(const Tensor& inputs) override {
        (void)inputs;
        return logitsParam.value;
    }

    void backward(const Tensor& gradOutput) override {
        logitsParam.grad = gradOutput;
    }

    std::vector<Parameter*> parameters() override {
        return { &logitsParam };
    }

    const Parameter& getParam() const {
        return logitsParam;
    }
};

void testTrainerBasicTrainingLoop() {
    DummyTrainableModel model;
    CrossEntropyLoss loss;
    SGDOptimizer optimizer(0.1f);

    TrainingConfig config;
    config.epochs = 3;
    config.logEverySteps = 1;
    config.checkpointEveryEpochs = 1;
    config.checkpointDirectory = ".";
    config.runName = "trainer_test";
    config.device = Device::CPU;

    Tensor inputs({ 1 }, 0.0f);

    Tensor targets({ 1 });
    targets[0] = 2.0f;

    TrainingBatch batch;
    batch.inputs = inputs;
    batch.targets = targets;

	std::cout << "==================================================\n";
	std::cout << "||            Trainer Basic Training Loop       ||\n";
	std::cout << "==================================================\n";

    std::vector<TrainingBatch> trainBatches = { batch };

    Trainer trainer(model, loss, optimizer, config);

    trainer.train(trainBatches);

    const TrainingHistory& history = trainer.getHistory();

    assert(history.trainLosses.size() == 3);

    assert(history.trainLosses[2] < history.trainLosses[0]);

    const Parameter& p = model.getParam();

    // Target class logit should move upward.
    assert(p.value[2] > 0.0f);

    // Non-target class logits should move downward.
    assert(p.value[0] < 0.0f);
    assert(p.value[1] < 0.0f);

    std::cout << "[PASS] Trainer basic training loop\n";
}

void testTensorToCUDAAndBack() {
    Tensor t({ 3 });
    t[0] = 1.0f;
    t[1] = 2.0f;
    t[2] = 3.0f;

    t.toCUDA();

    assert(t.hasDeviceData());
    assert(t.device() == Device::CUDA);

    t.toCPU();

    assert(near(t[0], 1.0f));
    assert(near(t[1], 2.0f));
    assert(near(t[2], 3.0f));

    std::cout << "[PASS] Tensor CUDA transfer\n";
}

void testTensorCudaCopyConstructor() {
    Tensor a({ 2 });
    a[0] = 4.0f;
    a[1] = 5.0f;

    a.toCUDA();

    Tensor b = a;
    b.toCPU();

    assert(near(b[0], 4.0f));
    assert(near(b[1], 5.0f));
    assert(b.hasDeviceData());

    std::cout << "[PASS] Tensor CUDA copy constructor\n";
}

void testTensorFillCUDA() {
    Tensor t({ 5 }, 0.0f);

    launchTensorFill(t, 3.5f);

    t.toCPU();

    for (size_t i = 0; i < t.size(); ++i) {
        assert(near(t[i], 3.5f));
    }

    std::cout << "[PASS] tensorFillCUDA\n";
}

void testTensorScaleCUDA() {
    Tensor t({ 4 });

    t[0] = 1.0f;
    t[1] = 2.0f;
    t[2] = -3.0f;
    t[3] = 0.5f;

    launchTensorScale(t, 2.0f);

    t.toCPU();

    assert(near(t[0], 2.0f));
    assert(near(t[1], 4.0f));
    assert(near(t[2], -6.0f));
    assert(near(t[3], 1.0f));

    std::cout << "[PASS] tensorScaleCUDA\n";
}

void testTensorAddCUDA() {
    Tensor a({ 4 });
    Tensor b({ 4 });
    Tensor out({ 4 });

    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;
    a[3] = 4.0f;

    b[0] = 10.0f;
    b[1] = 20.0f;
    b[2] = 30.0f;
    b[3] = 40.0f;

    launchTensorAdd(a, b, out);

    out.toCPU();

    assert(near(out[0], 11.0f));
    assert(near(out[1], 22.0f));
    assert(near(out[2], 33.0f));
    assert(near(out[3], 44.0f));

    std::cout << "[PASS] tensorAddCUDA\n";
}

void testSGDOptimizerCUDAParity() {
    Tensor cpuW({ 3 });
    cpuW[0] = 1.0f;
    cpuW[1] = 2.0f;
    cpuW[2] = -3.0f;

    Tensor gpuW = cpuW;

    Parameter cpuParam(cpuW, "cpu.w");
    Parameter gpuParam(gpuW, "gpu.w");

    cpuParam.grad[0] = 0.25f;
    cpuParam.grad[1] = 0.50f;
    cpuParam.grad[2] = -1.00f;

    gpuParam.grad[0] = 0.25f;
    gpuParam.grad[1] = 0.50f;
    gpuParam.grad[2] = -1.00f;

    gpuParam.value.toCUDA();
    gpuParam.grad.toCUDA();

    std::vector<Parameter*> cpuParams = { &cpuParam };
    std::vector<Parameter*> gpuParams = { &gpuParam };

    SGDOptimizer cpuOptimizer(0.1f, 0.01f);
    SGDOptimizer gpuOptimizer(0.1f, 0.01f);

    cpuOptimizer.step(cpuParams);
    gpuOptimizer.step(gpuParams);

    gpuParam.value.toCPU();

    assert(near(cpuParam.value[0], gpuParam.value[0]));
    assert(near(cpuParam.value[1], gpuParam.value[1]));
    assert(near(cpuParam.value[2], gpuParam.value[2]));

    std::cout << "[PASS] SGD CUDA parity\n";
}

void testAdamOptimizerCUDAParity() {
    Tensor cpuW({ 3 });
    cpuW[0] = 1.0f;
    cpuW[1] = 2.0f;
    cpuW[2] = -3.0f;

    Tensor gpuW = cpuW;

    Parameter cpuParam(cpuW, "cpu.adam.w");
    Parameter gpuParam(gpuW, "gpu.adam.w");

    std::vector<Parameter*> cpuParams = { &cpuParam };
    std::vector<Parameter*> gpuParams = { &gpuParam };

    cpuParam.grad[0] = 0.25f;
    cpuParam.grad[1] = 0.50f;
    cpuParam.grad[2] = -1.00f;

    gpuParam.grad[0] = 0.25f;
    gpuParam.grad[1] = 0.50f;
    gpuParam.grad[2] = -1.00f;

    gpuParam.value.toCUDA();
    gpuParam.grad.toCUDA();

    AdamOptimizer cpuOptimizer(
        0.001f,
        0.9f,
        0.999f,
        1e-8f,
        0.01f
    );

    AdamOptimizer gpuOptimizer(
        0.001f,
        0.9f,
        0.999f,
        1e-8f,
        0.01f
    );

    cpuOptimizer.step(cpuParams);
    gpuOptimizer.step(gpuParams);

    gpuParam.value.toCPU();

    assert(near(cpuParam.value[0], gpuParam.value[0], 1e-5f));
    assert(near(cpuParam.value[1], gpuParam.value[1], 1e-5f));
    assert(near(cpuParam.value[2], gpuParam.value[2], 1e-5f));

    std::cout << "[PASS] Adam CUDA parity single step\n";
}

void testAdamOptimizerCUDAParityMultipleSteps() {
    Tensor cpuW({ 2 });
    cpuW[0] = 1.0f;
    cpuW[1] = -2.0f;

    Tensor gpuW = cpuW;

    Parameter cpuParam(cpuW, "cpu.adam.multi.w");
    Parameter gpuParam(gpuW, "gpu.adam.multi.w");

    std::vector<Parameter*> cpuParams = { &cpuParam };
    std::vector<Parameter*> gpuParams = { &gpuParam };

    gpuParam.value.toCUDA();
    gpuParam.grad.toCUDA();

    AdamOptimizer cpuOptimizer(0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
    AdamOptimizer gpuOptimizer(0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);

    for (int step = 0; step < 5; ++step) {
        cpuParam.grad[0] = 0.25f + 0.01f * step;
        cpuParam.grad[1] = -0.50f + 0.02f * step;

        gpuParam.grad.toCPU();
        gpuParam.grad[0] = cpuParam.grad[0];
        gpuParam.grad[1] = cpuParam.grad[1];
        gpuParam.grad.toCUDA();

        cpuOptimizer.step(cpuParams);
        gpuOptimizer.step(gpuParams);
    }

    gpuParam.value.toCPU();

    assert(near(cpuParam.value[0], gpuParam.value[0], 1e-5f));
    assert(near(cpuParam.value[1], gpuParam.value[1], 1e-5f));

    std::cout << "[PASS] Adam CUDA parity multiple steps\n";
}

void testCrossEntropyLossCUDAParity() {
    Tensor cpuLogits({ 2, 4 });

    cpuLogits[0] = 1.0f;
    cpuLogits[1] = 2.0f;
    cpuLogits[2] = 3.0f;
    cpuLogits[3] = 4.0f;

    cpuLogits[4] = 0.5f;
    cpuLogits[5] = -1.0f;
    cpuLogits[6] = 2.0f;
    cpuLogits[7] = 0.0f;

    Tensor cpuTargets({ 2 });
    cpuTargets[0] = 3.0f;
    cpuTargets[1] = 2.0f;

    Tensor gpuLogits = cpuLogits;
    Tensor gpuTargets = cpuTargets;

    gpuLogits.toCUDA();
    gpuTargets.toCUDA();

    CrossEntropyLoss cpuLoss;
    CrossEntropyLoss gpuLoss;

    float cpuValue = cpuLoss.forward(cpuLogits, cpuTargets);
    Tensor cpuGrad = cpuLoss.backward();

    float gpuValue = gpuLoss.forward(gpuLogits, gpuTargets);
    Tensor gpuGrad = gpuLoss.backward();
    gpuGrad.toCPU();

    assert(near(cpuValue, gpuValue, 1e-5f));

    assert(cpuGrad.size() == gpuGrad.size());

    for (size_t i = 0; i < cpuGrad.size(); ++i) {
        assert(near(cpuGrad[i], gpuGrad[i], 1e-5f));
    }

    std::cout << "[PASS] CrossEntropyLoss CUDA parity\n";
}

void testCrossEntropyLossCUDAUniformParity() {
    Tensor cpuLogits({ 3, 5 }, 0.0f);

    Tensor cpuTargets({ 3 });
    cpuTargets[0] = 0.0f;
    cpuTargets[1] = 2.0f;
    cpuTargets[2] = 4.0f;

    Tensor gpuLogits = cpuLogits;
    Tensor gpuTargets = cpuTargets;

    gpuLogits.toCUDA();
    gpuTargets.toCUDA();

    CrossEntropyLoss cpuLoss;
    CrossEntropyLoss gpuLoss;

    float cpuValue = cpuLoss.forward(cpuLogits, cpuTargets);
    Tensor cpuGrad = cpuLoss.backward();

    float gpuValue = gpuLoss.forward(gpuLogits, gpuTargets);
    Tensor gpuGrad = gpuLoss.backward();
    gpuGrad.toCPU();

    assert(near(cpuValue, gpuValue, 1e-5f));

    for (size_t i = 0; i < cpuGrad.size(); ++i) {
        assert(near(cpuGrad[i], gpuGrad[i], 1e-5f));
    }

    std::cout << "[PASS] CrossEntropyLoss CUDA uniform parity\n";
}

void testLinearForwardCUDAParity() {
    Tensor input({ 2, 3 });
    input[0] = 1.0f; input[1] = 2.0f; input[2] = 3.0f;
    input[3] = 4.0f; input[4] = 5.0f; input[5] = 6.0f;

    Tensor weights({ 2, 3 });

    // output feature 0 weights
    weights[0] = 0.1f;
    weights[1] = 0.3f;
    weights[2] = 0.5f;

    // output feature 1 weights
    weights[3] = 0.2f;
    weights[4] = 0.4f;
    weights[5] = 0.6f;

    Tensor bias({ 2 });
    bias[0] = 0.5f;
    bias[1] = -0.5f;

    Tensor output({ 2, 2 }, 0.0f);

    input.toCUDA();
    weights.toCUDA();
    bias.toCUDA();
    output.toCUDA();

    launchLinearForward(
        input.deviceData(),
        weights.deviceData(),
        bias.deviceData(),
        output.deviceData(),
        2, 3, 2
    );

    output.toCPU();

    assert(near(output[0], 1.0f * 0.1f + 2.0f * 0.3f + 3.0f * 0.5f + 0.5f));
    assert(near(output[1], 1.0f * 0.2f + 2.0f * 0.4f + 3.0f * 0.6f - 0.5f));
    assert(near(output[2], 4.0f * 0.1f + 5.0f * 0.3f + 6.0f * 0.5f + 0.5f));
    assert(near(output[3], 4.0f * 0.2f + 5.0f * 0.4f + 6.0f * 0.6f - 0.5f));

    std::cout << "[PASS] Linear forward CUDA parity\n";
}

void testLinearClassForwardCUDAParity() {
    Random rng(42);

    Linear cpuLinear(3, 2, rng);
    Linear gpuLinear = cpuLinear;

    Tensor cpuInput({ 2, 3 });
    cpuInput[0] = 1.0f;
    cpuInput[1] = 2.0f;
    cpuInput[2] = 3.0f;
    cpuInput[3] = 4.0f;
    cpuInput[4] = 5.0f;
    cpuInput[5] = 6.0f;

    Tensor gpuInput = cpuInput;
    gpuInput.toCUDA();

    Tensor cpuOutput = cpuLinear.forward(cpuInput);
    Tensor gpuOutput = gpuLinear.forward(gpuInput);

    gpuOutput.toCPU();

    assert(cpuOutput.size() == gpuOutput.size());

    for (size_t i = 0; i < cpuOutput.size(); ++i) {
        assert(near(cpuOutput[i], gpuOutput[i], 1e-5f));
    }

    std::cout << "[PASS] Linear::forward CUDA parity\n";
}

void testEmbeddingForwardCUDAParity() {
    Random rng(42);

    Embedding cpuEmbedding(5, 3, rng);
    Embedding gpuEmbedding = cpuEmbedding;

    Tensor tokenIds({ 2, 4 });
    tokenIds[0] = 1.0f;
    tokenIds[1] = 2.0f;
    tokenIds[2] = 1.0f;
    tokenIds[3] = 3.0f;

    tokenIds[4] = 0.0f;
    tokenIds[5] = 2.0f;
    tokenIds[6] = 2.0f;
    tokenIds[7] = 4.0f;

    Tensor cpuOutput = cpuEmbedding.forward(tokenIds);

    Tensor gpuTokenIds = tokenIds;
    gpuTokenIds.toCUDA();

    Tensor gpuOutput = gpuEmbedding.forward(gpuTokenIds);
    gpuOutput.toCPU();

    assert(cpuOutput.shape() == gpuOutput.shape());

    for (size_t i = 0; i < cpuOutput.size(); ++i) {
        assert(near(cpuOutput[i], gpuOutput[i], 1e-5f));
    }

    std::cout << "[PASS] Embedding forward CUDA parity\n";
}

void testLinearBackward() {
    Random rng(42);
    Linear linear(2, 3, rng);

    Tensor input({ 2, 2 });
    input[0] = 1.0f; input[1] = 2.0f;
    input[2] = 3.0f; input[3] = 4.0f;

    Tensor output = linear.forward(input);
    (void)output;

    Tensor gradOutput({ 2, 3 });
    gradOutput[0] = 0.1f; gradOutput[1] = 0.2f; gradOutput[2] = 0.3f;
    gradOutput[3] = 0.4f; gradOutput[4] = 0.5f; gradOutput[5] = 0.6f;

    Tensor gradInput = linear.backward(gradOutput);

    std::vector<Parameter*> params = linear.parameters();
    Parameter* weights = params[0];
    Parameter* bias = params[1];

    // bias grad = column sums of gradOutput
    assert(near(bias->grad[0], 0.1f + 0.4f));
    assert(near(bias->grad[1], 0.2f + 0.5f));
    assert(near(bias->grad[2], 0.3f + 0.6f));

    // weight grad[o, i] = sum_b gradOutput[b, o] * input[b, i]
    assert(near(weights->grad.at({ 0, 0 }), 0.1f * 1.0f + 0.4f * 3.0f));
    assert(near(weights->grad.at({ 0, 1 }), 0.1f * 2.0f + 0.4f * 4.0f));

    assert(near(weights->grad.at({ 1, 0 }), 0.2f * 1.0f + 0.5f * 3.0f));
    assert(near(weights->grad.at({ 1, 1 }), 0.2f * 2.0f + 0.5f * 4.0f));

    assert(near(weights->grad.at({ 2, 0 }), 0.3f * 1.0f + 0.6f * 3.0f));
    assert(near(weights->grad.at({ 2, 1 }), 0.3f * 2.0f + 0.6f * 4.0f));

    // gradInput[b, i] = sum_o gradOutput[b, o] * weight[o, i]
    const Tensor& w = linear.weights();

    assert(near(
        gradInput.at({ 0, 0 }),
        0.1f * w.at({ 0, 0 }) + 0.2f * w.at({ 1, 0 }) + 0.3f * w.at({ 2, 0 })
    ));

    assert(near(
        gradInput.at({ 0, 1 }),
        0.1f * w.at({ 0, 1 }) + 0.2f * w.at({ 1, 1 }) + 0.3f * w.at({ 2, 1 })
    ));

    assert(near(
        gradInput.at({ 1, 0 }),
        0.4f * w.at({ 0, 0 }) + 0.5f * w.at({ 1, 0 }) + 0.6f * w.at({ 2, 0 })
    ));

    assert(near(
        gradInput.at({ 1, 1 }),
        0.4f * w.at({ 0, 1 }) + 0.5f * w.at({ 1, 1 }) + 0.6f * w.at({ 2, 1 })
    ));

    std::cout << "[PASS] Linear backward\n";
}

void testFFNBackward() {
    Random rng(42);

    FFN ffn(
        4,      // embedDim
        8,      // hiddenDim
        rng
    );

    Tensor input({ 2, 3, 4 });

    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1) * 0.1f;
    }

    Tensor output = ffn.forward(input);

    Tensor gradOutput(output.shape(), 1.0f);

    Tensor gradInput = ffn.backward(gradOutput);

    // Shape checks
    assert(gradInput.shape() == input.shape());

    // Parameter gradients exist
    auto params = ffn.parameters();

    assert(params.size() == 4);

    for (Parameter* p : params) {
        bool foundNonZero = false;

        for (size_t i = 0; i < p->grad.size(); ++i) {
            if (std::fabs(p->grad[i]) > 1e-7f) {
                foundNonZero = true;
                break;
            }
        }

        assert(foundNonZero);
    }

    // Gradient isn't identically zero
    bool gradInputNonZero = false;

    for (size_t i = 0; i < gradInput.size(); ++i) {
        if (std::fabs(gradInput[i]) > 1e-7f) {
            gradInputNonZero = true;
            break;
        }
    }

    assert(gradInputNonZero);

    FFN ffn1(4, 8, rng);

    Tensor grad({ 2,3,4 }, 1.0f);

    bool threw = false;

    try {
        ffn1.backward(grad);
    }
    catch (...) {
        threw = true;
    }

    assert(threw);

    std::cout << "[PASS] FFN backward\n";
}

void testLayerNormBackward() {
    LayerNorm norm(4);

    Tensor input({ 2, 3, 4 });

    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1) * 0.1f;
    }

    Tensor output = norm.forward(input);
    Tensor gradOutput(output.shape(), 1.0f);

    Tensor gradInput = norm.backward(gradOutput);

    assert(gradInput.shape() == input.shape());

    auto params = norm.parameters();
    assert(params.size() == 2);

    Parameter* gamma = params[0];
    Parameter* beta = params[1];

    assert(gamma->grad.size() == 4);
    assert(beta->grad.size() == 4);

    for (size_t i = 0; i < beta->grad.size(); ++i) {
        assert(near(beta->grad[i], 6.0f));
    }

    std::cout << "[PASS] LayerNorm backward\n";
}

void testSelfAttentionBackward() {
    Random rng(42);

    constexpr size_t embedDim = 4;
    constexpr size_t batchSize = 2;
    constexpr size_t sequenceLength = 3;

    SelfAttention attention(embedDim, rng);

    Tensor input({ batchSize, sequenceLength, embedDim });

    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1) * 0.05f;
    }

    Tensor output = attention.forward(input);
    Tensor gradOutput(output.shape(), 1.0f);

    Tensor gradInput = attention.backward(gradOutput);

    assert(gradInput.shape() == input.shape());

    auto params = attention.parameters();
    assert(!params.empty());

    bool foundNonZeroParamGrad = false;

    for (Parameter* p : params) {
        for (size_t i = 0; i < p->grad.size(); ++i) {
            if (std::fabs(p->grad[i]) > 1e-7f) {
                foundNonZeroParamGrad = true;
                break;
            }
        }

        if (foundNonZeroParamGrad) {
            break;
        }
    }

    assert(foundNonZeroParamGrad);

    bool foundNonZeroInputGrad = false;

    for (size_t i = 0; i < gradInput.size(); ++i) {
        if (std::fabs(gradInput[i]) > 1e-7f) {
            foundNonZeroInputGrad = true;
            break;
        }
    }

    assert(foundNonZeroInputGrad);

    std::cout << "[PASS] SelfAttention backward\n";
}

void testTransformerBlockSingleHeadBackward() {
    Random rng(42);

    TransformerBlockConfig config;
    config.d_model = 4;
    config.d_ff = 8;
    config.pre_norm = true;
    config.use_bias = true;
    config.residual_dropout = 0.0f;
    config.ffn_dropout = 0.0f;
    config.attentionType = AttentionType::SingleHead;
    config.numHeads = 1;

    config.validate();

    TransformerBlock block(config, rng);

    Tensor input({ 2, 3, 4 });

    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1) * 0.05f;
    }

    Tensor output = block.forward(input);
    Tensor gradOutput(output.shape(), 1.0f);

    Tensor gradInput = block.backward(gradOutput);

    assert(gradInput.shape() == input.shape());

    auto params = block.parameters();
    assert(!params.empty());

    bool foundNonZeroParamGrad = false;

    for (Parameter* p : params) {
        for (size_t i = 0; i < p->grad.size(); ++i) {
            if (std::fabs(p->grad[i]) > 1e-7f) {
                foundNonZeroParamGrad = true;
                break;
            }
        }

        if (foundNonZeroParamGrad) {
            break;
        }
    }

    assert(foundNonZeroParamGrad);

    std::cout << "[PASS] TransformerBlock single-head backward\n";
}

void testMultiHeadAttentionBackward() {
    Random rng(42);

    constexpr size_t embedDim = 8;
    constexpr size_t numHeads = 2;
    constexpr size_t batchSize = 2;
    constexpr size_t sequenceLength = 3;
	AttentionConfig config = { embedDim, numHeads, true, true, 0.0f };

    MultiHeadAttention attention(config, rng);

    Tensor input({ batchSize, sequenceLength, embedDim });

    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1) * 0.025f;
    }

    Tensor output = attention.forward(input);
    Tensor gradOutput(output.shape(), 1.0f);

    Tensor gradInput = attention.backward(gradOutput);

    assert(gradInput.shape() == input.shape());

    auto params = attention.parameters();
    assert(!params.empty());

    bool foundNonZeroParamGrad = false;

    for (Parameter* p : params) {
        for (size_t i = 0; i < p->grad.size(); ++i) {
            if (std::fabs(p->grad[i]) > 1e-7f) {
                foundNonZeroParamGrad = true;
                break;
            }
        }

        if (foundNonZeroParamGrad) {
            break;
        }
    }

    assert(foundNonZeroParamGrad);

    bool foundNonZeroInputGrad = false;

    for (size_t i = 0; i < gradInput.size(); ++i) {
        if (std::fabs(gradInput[i]) > 1e-7f) {
            foundNonZeroInputGrad = true;
            break;
        }
    }

    assert(foundNonZeroInputGrad);

    std::cout << "[PASS] MultiHeadAttention backward\n";
}


void testTransformerBlockMultiHeadBackward() {
    Random rng(42);

    TransformerBlockConfig config;
    config.d_model = 8;
    config.d_ff = 16;
    config.pre_norm = true;
    config.use_bias = true;
    config.residual_dropout = 0.0f;
    config.ffn_dropout = 0.0f;
    config.attentionType = AttentionType::MultiHead;
    config.numHeads = 2;

    config.validate();

    TransformerBlock block(config, rng);

    Tensor input({ 2, 3, 8 });

    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1) * 0.025f;
    }

    Tensor output = block.forward(input);
    Tensor gradOutput(output.shape(), 1.0f);

    Tensor gradInput = block.backward(gradOutput);

    assert(gradInput.shape() == input.shape());

    auto params = block.parameters();
    assert(!params.empty());

    bool foundNonZeroParamGrad = false;

    for (Parameter* p : params) {
        for (size_t i = 0; i < p->grad.size(); ++i) {
            if (std::fabs(p->grad[i]) > 1e-7f) {
                foundNonZeroParamGrad = true;
                break;
            }
        }

        if (foundNonZeroParamGrad) {
            break;
        }
    }

    assert(foundNonZeroParamGrad);

    std::cout << "[PASS] TransformerBlock multi-head backward\n";
}

void testEmbeddingBackward() {
    Random rng(42);

    Embedding embedding(
        5,  // vocabSize
        3,  // embeddingDim
        rng
    );

    Tensor tokenIds({ 2, 4 });

    tokenIds[0] = 1.0f;
    tokenIds[1] = 2.0f;
    tokenIds[2] = 1.0f;
    tokenIds[3] = 3.0f;

    tokenIds[4] = 0.0f;
    tokenIds[5] = 2.0f;
    tokenIds[6] = 2.0f;
    tokenIds[7] = 4.0f;

    Tensor output = embedding.forward(tokenIds);

    Tensor gradOutput(output.shape(), 1.0f);

    Tensor gradInput = embedding.backward(gradOutput);

    assert(gradInput.shape() == tokenIds.shape());

    auto params = embedding.parameters();
    assert(params.size() == 1);

    Parameter* table = params[0];

    assert(table->grad.shape()[0] == 5);
    assert(table->grad.shape()[1] == 3);

    // token 0 appears once
    for (size_t e = 0; e < 3; ++e) {
        assert(near(table->grad.at({ 0, e }), 1.0f));
    }

    // token 1 appears twice
    for (size_t e = 0; e < 3; ++e) {
        assert(near(table->grad.at({ 1, e }), 2.0f));
    }

    // token 2 appears three times
    for (size_t e = 0; e < 3; ++e) {
        assert(near(table->grad.at({ 2, e }), 3.0f));
    }

    // token 3 appears once
    for (size_t e = 0; e < 3; ++e) {
        assert(near(table->grad.at({ 3, e }), 1.0f));
    }

    // token 4 appears once
    for (size_t e = 0; e < 3; ++e) {
        assert(near(table->grad.at({ 4, e }), 1.0f));
    }

    std::cout << "[PASS] Embedding backward\n";
}

void testTransformerBackwardSingleHead() {
    Random rng(42);

    TransformerBlockConfig blockConfig;
    blockConfig.d_model = 8;
    blockConfig.d_ff = 16;
    blockConfig.pre_norm = true;
    blockConfig.use_bias = true;
    blockConfig.residual_dropout = 0.0f;
    blockConfig.ffn_dropout = 0.0f;
    blockConfig.attentionType = AttentionType::SingleHead;
    blockConfig.numHeads = 1;

    TransformerModelConfig modelConfig;
    modelConfig.vocab_size = 12;
    modelConfig.max_seq_len = 5;
    modelConfig.num_layers = 1;
    modelConfig.block = blockConfig;
    modelConfig.learned_positional_embeddings = true;
    modelConfig.embedding_dropout = 0.0f;

    modelConfig.validate();

    Transformer model(modelConfig, rng);

    Tensor inputs({ 2, 5 });
    Tensor targets({ 2, 5 });

    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs[i] = static_cast<float>(i % modelConfig.vocab_size);
        targets[i] = static_cast<float>((i + 1) % modelConfig.vocab_size);
    }

    Tensor logits = model.forward(inputs);

    // CrossEntropyLoss currently expects [batch, classes],
    // so flatten [B, T, V] -> [B*T, V], targets [B,T] -> [B*T].
    Tensor flatLogits =
        LayerUtils::flatten3DTo2D(logits);

    Tensor flatTargets({ targets.size() });

    for (size_t i = 0; i < targets.size(); ++i) {
        flatTargets[i] = targets[i];
    }

    CrossEntropyLoss loss;
    float lossValue = loss.forward(flatLogits, flatTargets);

    assert(std::isfinite(lossValue));

    Tensor flatGrad = loss.backward();

    Tensor gradLogits =
        LayerUtils::unflatten2DTo3D(
            flatGrad,
            2,
            5
        );

    model.backward(gradLogits);

    auto params = model.parameters();
    assert(!params.empty());

    bool foundNonZeroGrad = false;

    for (Parameter* p : params) {
        for (size_t i = 0; i < p->grad.size(); ++i) {
            if (std::fabs(p->grad[i]) > 1e-7f) {
                foundNonZeroGrad = true;
                break;
            }
        }

        if (foundNonZeroGrad) {
            break;
        }
    }

    assert(foundNonZeroGrad);

    std::cout << "[PASS] Transformer backward single-head\n";
}

void testTransformerBackwardMultiHead() {
    Random rng(42);

    TransformerBlockConfig blockConfig;
    blockConfig.d_model = 8;
    blockConfig.d_ff = 16;
    blockConfig.pre_norm = true;
    blockConfig.use_bias = true;
    blockConfig.residual_dropout = 0.0f;
    blockConfig.ffn_dropout = 0.0f;
    blockConfig.attentionType = AttentionType::MultiHead;
    blockConfig.numHeads = 4;

    TransformerModelConfig modelConfig;
    modelConfig.vocab_size = 12;
    modelConfig.max_seq_len = 5;
    modelConfig.num_layers = 1;
    modelConfig.block = blockConfig;
    modelConfig.learned_positional_embeddings = true;
    modelConfig.embedding_dropout = 0.0f;

    modelConfig.validate();

    Transformer model(modelConfig, rng);

    Tensor inputs({ 2, 5 });
    Tensor targets({ 2, 5 });

    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs[i] = static_cast<float>(i % modelConfig.vocab_size);
        targets[i] = static_cast<float>((i + 1) % modelConfig.vocab_size);
    }

    Tensor logits = model.forward(inputs);

    // CrossEntropyLoss currently expects [batch, classes],
    // so flatten [B, T, V] -> [B*T, V], targets [B,T] -> [B*T].
    Tensor flatLogits =
        LayerUtils::flatten3DTo2D(logits);

    Tensor flatTargets({ targets.size() });

    for (size_t i = 0; i < targets.size(); ++i) {
        flatTargets[i] = targets[i];
    }

    CrossEntropyLoss loss;
    float lossValue = loss.forward(flatLogits, flatTargets);

    assert(std::isfinite(lossValue));

    Tensor flatGrad = loss.backward();

    Tensor gradLogits =
        LayerUtils::unflatten2DTo3D(
            flatGrad,
            2,
            5
        );

    model.backward(gradLogits);

    auto params = model.parameters();
    assert(!params.empty());

    bool foundNonZeroGrad = false;

    for (Parameter* p : params) {
        for (size_t i = 0; i < p->grad.size(); ++i) {
            if (std::fabs(p->grad[i]) > 1e-7f) {
                foundNonZeroGrad = true;
                break;
            }
        }

        if (foundNonZeroGrad) {
            break;
        }
    }

    assert(foundNonZeroGrad);

    std::cout << "[PASS] Transformer backward multi-head\n";
}

void testLinearBackwardCUDAParity() {

    Random rng(42);

    Linear cpuLinear(3, 2, rng);
    Linear gpuLinear = cpuLinear;

    Tensor input({ 4, 3 });
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1) * 0.1f;
    }

    Tensor gradOutput({ 4, 2 });
    for (size_t i = 0; i < gradOutput.size(); ++i) {
        gradOutput[i] = static_cast<float>(i + 1) * 0.05f;
    }

    Tensor cpuOutput = cpuLinear.forward(input);
    (void)cpuOutput;

    Tensor cpuGradInput = cpuLinear.backward(gradOutput);

    Tensor gpuInput = input;
    Tensor gpuGradOutput = gradOutput;

    gpuInput.toCUDA();
    gpuGradOutput.toCUDA();

    Tensor gpuOutput = gpuLinear.forward(gpuInput);
    (void)gpuOutput;

    Tensor gpuGradInput = gpuLinear.backward(gpuGradOutput);

    gpuGradInput.toCPU();

    auto cpuParams = cpuLinear.parameters();
    auto gpuParams = gpuLinear.parameters();

    Parameter* cpuWeights = cpuParams[0];
    Parameter* cpuBias = cpuParams[1];

    Parameter* gpuWeights = gpuParams[0];
    Parameter* gpuBias = gpuParams[1];

    gpuWeights->grad.toCPU();
    gpuBias->grad.toCPU();

    assert(cpuGradInput.shape() == gpuGradInput.shape());

    for (size_t i = 0; i < cpuGradInput.size(); ++i) {
        assert(near(cpuGradInput[i], gpuGradInput[i], 1e-5f));
    }

    for (size_t i = 0; i < cpuWeights->grad.size(); ++i) {
        assert(near(cpuWeights->grad[i], gpuWeights->grad[i], 1e-5f));
    }

    for (size_t i = 0; i < cpuBias->grad.size(); ++i) {
        assert(near(cpuBias->grad[i], gpuBias->grad[i], 1e-5f));
    }

    std::cout << "[PASS] Linear backward CUDA parity\n";
}

void testEmbeddingBackwardCUDAParity() {
    Random rng(42);

    Embedding cpuEmbedding(5, 3, rng);
    Embedding gpuEmbedding = cpuEmbedding;

    Tensor tokenIds({ 2, 4 });
    tokenIds[0] = 1.0f;
    tokenIds[1] = 2.0f;
    tokenIds[2] = 1.0f;
    tokenIds[3] = 3.0f;
    tokenIds[4] = 0.0f;
    tokenIds[5] = 2.0f;
    tokenIds[6] = 2.0f;
    tokenIds[7] = 4.0f;

    Tensor cpuOutput = cpuEmbedding.forward(tokenIds);
    Tensor cpuGradOutput(cpuOutput.shape(), 1.0f);
    cpuEmbedding.backward(cpuGradOutput);

    Tensor gpuTokenIds = tokenIds;
    gpuTokenIds.toCUDA();

    Tensor gpuOutput = gpuEmbedding.forward(gpuTokenIds);
    Tensor gpuGradOutput(gpuOutput.shape(), 1.0f);
    gpuGradOutput.toCUDA();

    gpuEmbedding.backward(gpuGradOutput);

    auto cpuParams = cpuEmbedding.parameters();
    auto gpuParams = gpuEmbedding.parameters();

    Parameter* cpuTable = cpuParams[0];
    Parameter* gpuTable = gpuParams[0];

    gpuTable->grad.toCPU();

    for (size_t i = 0; i < cpuTable->grad.size(); ++i) {
        assert(near(cpuTable->grad[i], gpuTable->grad[i], 1e-5f));
    }

    std::cout << "[PASS] Embedding backward CUDA parity\n";
}

void testLayerNormForwardCUDAParity() {
    LayerNorm cpuNorm(4);
    LayerNorm gpuNorm(4);

    Tensor input({ 2, 3, 4 });

    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1) * 0.1f;
    }

    Tensor cpuOutput = cpuNorm.forward(input);

    Tensor gpuInput = input;
    gpuInput.toCUDA();

    Tensor gpuOutput = gpuNorm.forward(gpuInput);
    gpuOutput.toCPU();

    assert(cpuOutput.shape() == gpuOutput.shape());

    for (size_t i = 0; i < cpuOutput.size(); ++i) {
        assert(near(cpuOutput[i], gpuOutput[i], 1e-5f));
    }

    std::cout << "[PASS] LayerNorm forward CUDA parity\n";
}

void testLayerNormBackwardCUDAParity() {
    LayerNorm cpuNorm(4);
    LayerNorm gpuNorm(4);

    Tensor input({ 2, 3, 4 });

    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1) * 0.1f;
    }

    Tensor gradOutput({ 2, 3, 4 });

    for (size_t i = 0; i < gradOutput.size(); ++i) {
        gradOutput[i] = static_cast<float>(i + 1) * 0.05f;
    }

    Tensor cpuOutput = cpuNorm.forward(input);
    (void)cpuOutput;

    Tensor cpuGradInput = cpuNorm.backward(gradOutput);

    Tensor gpuInput = input;
    Tensor gpuGradOutput = gradOutput;

    gpuInput.toCUDA();
    gpuGradOutput.toCUDA();

    Tensor gpuOutput = gpuNorm.forward(gpuInput);
    (void)gpuOutput;

    Tensor gpuGradInput = gpuNorm.backward(gpuGradOutput);

    gpuGradInput.toCPU();

    auto cpuParams = cpuNorm.parameters();
    auto gpuParams = gpuNorm.parameters();

    Parameter* cpuGamma = cpuParams[0];
    Parameter* cpuBeta = cpuParams[1];

    Parameter* gpuGamma = gpuParams[0];
    Parameter* gpuBeta = gpuParams[1];

    gpuGamma->grad.toCPU();
    gpuBeta->grad.toCPU();

    assert(cpuGradInput.shape() == gpuGradInput.shape());

    for (size_t i = 0; i < cpuGradInput.size(); ++i) {
        assert(near(cpuGradInput[i], gpuGradInput[i], 1e-5f));
    }

    for (size_t i = 0; i < cpuGamma->grad.size(); ++i) {
        assert(near(cpuGamma->grad[i], gpuGamma->grad[i], 1e-5f));
    }

    for (size_t i = 0; i < cpuBeta->grad.size(); ++i) {
        assert(near(cpuBeta->grad[i], gpuBeta->grad[i], 1e-5f));
    }

    std::cout << "[PASS] LayerNorm backward CUDA parity\n";
}

void testGeluCUDAParity() {
    Tensor cpu({ 6 });
    cpu[0] = -2.0f;
    cpu[1] = -1.0f;
    cpu[2] = -0.5f;
    cpu[3] = 0.0f;
    cpu[4] = 0.5f;
    cpu[5] = 2.0f;

    Tensor gpu = cpu;
    gpu.toCUDA();

    for (size_t i = 0; i < cpu.size(); ++i) {
        cpu[i] = MathUtils::gelu(cpu[i]);
    }

    launchGeluForward(gpu.deviceData(), gpu.size());
    gpu.toCPU();

    for (size_t i = 0; i < cpu.size(); ++i) {
        assert(near(cpu[i], gpu[i], 1e-5f));
    }

    std::cout << "[PASS] GELU forward CUDA parity\n";
}

void testGeluBackwardCUDAParity() {
    Tensor pre({ 6 });
    pre[0] = -2.0f;
    pre[1] = -1.0f;
    pre[2] = -0.5f;
    pre[3] = 0.0f;
    pre[4] = 0.5f;
    pre[5] = 2.0f;

    Tensor cpuGrad({ 6 }, 1.0f);
    Tensor gpuPre = pre;
    Tensor gpuGrad = cpuGrad;

    gpuPre.toCUDA();
    gpuGrad.toCUDA();

    for (size_t i = 0; i < cpuGrad.size(); ++i) {
        cpuGrad[i] *= MathUtils::geluDerivative(pre[i]);
    }

    launchGeluBackward(
        gpuPre.deviceData(),
        gpuGrad.deviceData(),
        gpuGrad.size()
    );

    gpuGrad.toCPU();

    for (size_t i = 0; i < cpuGrad.size(); ++i) {
        assert(near(cpuGrad[i], gpuGrad[i], 1e-5f));
    }

    std::cout << "[PASS] GELU backward CUDA parity\n";
}

void testSelfAttentionForwardCUDAParity() {
    Random rng(42);

    constexpr size_t embedDim = 4;
    constexpr size_t batchSize = 2;
    constexpr size_t sequenceLength = 3;

    SelfAttention cpuAttention(embedDim, rng);
    SelfAttention gpuAttention = cpuAttention;

    Tensor input({ batchSize, sequenceLength, embedDim });

    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1) * 0.05f;
    }

    Tensor cpuOutput = cpuAttention.forward(input);

    Tensor gpuInput = input;
    gpuInput.toCUDA();

    Tensor gpuOutput = gpuAttention.forward(gpuInput);
    gpuOutput.toCPU();

    assert(cpuOutput.shape() == gpuOutput.shape());

    for (size_t i = 0; i < cpuOutput.size(); ++i) {
        assert(near(cpuOutput[i], gpuOutput[i], 1e-5f));
    }

    std::cout << "[PASS] SelfAttention forward CUDA parity\n";
}

void testMultiHeadAttentionForwardCUDAParity() {
    Random rng(42);

    constexpr size_t embedDim = 4;
    constexpr size_t batchSize = 2;
    constexpr size_t sequenceLength = 3;

    MultiHeadAttention cpuAttention({ embedDim, 2 }, rng); // 2 heads
    MultiHeadAttention gpuAttention = cpuAttention;

    Tensor input({ batchSize, sequenceLength, embedDim });

    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1) * 0.05f;
    }

    Tensor cpuOutput = cpuAttention.forward(input);

    Tensor gpuInput = input;
    gpuInput.toCUDA();

    Tensor gpuOutput = gpuAttention.forward(gpuInput);
    gpuOutput.toCPU();

    assert(cpuOutput.shape() == gpuOutput.shape());

    for (size_t i = 0; i < cpuOutput.size(); ++i) {
        assert(near(cpuOutput[i], gpuOutput[i], 1e-5f));
    }

    std::cout << "[PASS] MultiHeadAttention forward CUDA parity\n";
}

void testSelfAttentionBackwardCUDAParity() {
    Random rng(42);

    constexpr size_t embedDim = 4;
    constexpr size_t batchSize = 2;
    constexpr size_t sequenceLength = 3;

    SelfAttention cpuAttention(embedDim, rng);
    SelfAttention gpuAttention = cpuAttention;

    Tensor input({ batchSize, sequenceLength, embedDim });

    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1) * 0.05f;
    }

    Tensor gradOutput({ batchSize, sequenceLength, embedDim }, 1.0f);

    Tensor cpuOutput = cpuAttention.forward(input);
    (void)cpuOutput;

    Tensor cpuGradInput = cpuAttention.backward(gradOutput);

    Tensor gpuInput = input;
    Tensor gpuGradOutput = gradOutput;

    gpuInput.toCUDA();
    gpuGradOutput.toCUDA();

    Tensor gpuOutput = gpuAttention.forward(gpuInput);
    (void)gpuOutput;

    Tensor gpuGradInput = gpuAttention.backward(gpuGradOutput);
    gpuGradInput.toCPU();

    assert(cpuGradInput.shape() == gpuGradInput.shape());

    for (size_t i = 0; i < cpuGradInput.size(); ++i) {
        assert(near(cpuGradInput[i], gpuGradInput[i], 1e-4f));
    }

    auto cpuParams = cpuAttention.parameters();
    auto gpuParams = gpuAttention.parameters();

    assert(cpuParams.size() == gpuParams.size());

    for (size_t p = 0; p < cpuParams.size(); ++p) {
        gpuParams[p]->grad.toCPU();

        for (size_t i = 0; i < cpuParams[p]->grad.size(); ++i) {
            assert(near(cpuParams[p]->grad[i], gpuParams[p]->grad[i], 1e-4f));
        }
    }

    std::cout << "[PASS] SelfAttention backward CUDA parity\n";
}

void testMultiHeadAttentionBackwardCUDAParity() {
    Random rng(42);

    constexpr size_t embedDim = 4;
    constexpr size_t batchSize = 2;
    constexpr size_t sequenceLength = 3;
    constexpr size_t heads = 2;

    AttentionConfig config(embedDim, heads);

    MultiHeadAttention cpuAttention(config, rng);
    MultiHeadAttention gpuAttention = cpuAttention;

    Tensor input({ batchSize, sequenceLength, embedDim });

    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1) * 0.05f;
    }

    Tensor gradOutput({ batchSize, sequenceLength, embedDim }, 1.0f);

    Tensor cpuOutput = cpuAttention.forward(input);
    (void)cpuOutput;

    Tensor cpuGradInput = cpuAttention.backward(gradOutput);

    Tensor gpuInput = input;
    Tensor gpuGradOutput = gradOutput;

    gpuInput.toCUDA();
    gpuGradOutput.toCUDA();

    Tensor gpuOutput = gpuAttention.forward(gpuInput);
    (void)gpuOutput;

    Tensor gpuGradInput = gpuAttention.backward(gpuGradOutput);
    gpuGradInput.toCPU();

    assert(cpuGradInput.shape() == gpuGradInput.shape());

    for (size_t i = 0; i < cpuGradInput.size(); ++i) {
        assert(near(cpuGradInput[i], gpuGradInput[i], 1e-4f));
    }

    auto cpuParams = cpuAttention.parameters();
    auto gpuParams = gpuAttention.parameters();

    assert(cpuParams.size() == gpuParams.size());

    for (size_t p = 0; p < cpuParams.size(); ++p) {
        gpuParams[p]->grad.toCPU();

        for (size_t i = 0; i < cpuParams[p]->grad.size(); ++i) {
            assert(near(cpuParams[p]->grad[i], gpuParams[p]->grad[i], 1e-4f));
        }
    }

    std::cout << "[PASS] MultiHeadAttention backward CUDA parity\n";
}

void testEpochCallback() {
    DummyTrainableModel model;
    CrossEntropyLoss loss;
    SGDOptimizer optimizer(0.1f);

    TrainingConfig config;
    config.epochs = 2;
    config.logEverySteps = 1;
    config.checkpointEveryEpochs = 1;
    config.checkpointDirectory = ".";
    config.runName = "epoch_callback";

    CheckpointCallback callback(config.checkpointDirectory, config.checkpointEveryEpochs);

    Tensor inputs({ 1, 1 }, 0.0f);

    Tensor targets({ 1 });
    targets[0] = 2.0f;

    TrainingBatch batch;
    batch.inputs = inputs;
    batch.targets = targets;

    std::vector<TrainingBatch> trainBatches = { batch };

    Trainer trainer(model, loss, optimizer, config);

    trainer.addCallback(&callback);

    trainer.train(trainBatches);

    const TrainingHistory& history = trainer.getHistory();

    assert(history.trainLosses.size() == 2);

    assert(history.trainLosses[1] < history.trainLosses[0]);

    std::cout << "[PASS] Epoch Callback\n";
}

void testBestCheckpointCallback() {
    DummyTrainableModel model;
    CrossEntropyLoss loss;
    SGDOptimizer optimizer(0.1f);

    TrainingConfig config;
    config.epochs = 3;
    config.logEverySteps = 1;
    config.checkpointDirectory = ".";
    config.runName = "best_checkpoint_test";
    config.device = Device::CPU;

    Tensor inputs({ 1, 1 }, 0.0f);

    Tensor targets({ 1 });
    targets[0] = 2.0f;

    TrainingBatch batch;
    batch.inputs = inputs;
    batch.targets = targets;

    std::vector<TrainingBatch> trainBatches = { batch };

    Trainer trainer(model, loss, optimizer, config);

    BestCheckpointCallback bestCallback(
        config.checkpointDirectory,
        0.0f,
        false
    );

    trainer.addCallback(&bestCallback);

    trainer.train(trainBatches);

    assert(bestCallback.hasBestCheckpoint());
    assert(bestCallback.getBestEpoch() == 3);
    assert(bestCallback.getBestMetric() == trainer.getHistory().trainLosses.back());

    std::vector<Parameter*> params = model.parameters();

    params[0]->value[0] = 999.0f;
    params[0]->value[1] = 999.0f;
    params[0]->value[2] = -999.0f;

    CheckpointMetadata metadata;
    TrainingHistory loadedHistory;

    Checkpoint::load(
        bestCallback.getBestCheckpointPath(),
        params,
        metadata,
        loadedHistory
    );

    assert(metadata.epoch == 3);
    assert(metadata.runName == config.runName);

    assert(params[0]->value[2] > 0.0f);
    assert(params[0]->value[0] < 0.0f);
    assert(params[0]->value[1] < 0.0f);

    assert(loadedHistory.trainLosses.size() == 3);

    std::cout << "[PASS] BestCheckpointCallback\n";
}

void testEarlyStoppingCallback() {
    DummyTrainableModel model;
    CrossEntropyLoss loss;
    SGDOptimizer optimizer(0.000001f);

    TrainingConfig config;
    config.epochs = 10;
    config.logEverySteps = 0;
    config.checkpointDirectory = ".";
    config.runName = "early_stopping_test";
    config.device = Device::CPU;

    Tensor inputs({ 1, 1 }, 0.0f);

    Tensor targets({ 1 });
    targets[0] = 2.0f;

    TrainingBatch batch;
    batch.inputs = inputs;
    batch.targets = targets;

    std::vector<TrainingBatch> trainBatches = { batch };

    Trainer trainer(model, loss, optimizer, config);

    EarlyStoppingCallback earlyStop(
        2,       // patience
        100.0f,  // huge minDelta means only epoch 1 counts as improvement
        false    // use training loss
    );

    trainer.addCallback(&earlyStop);

    trainer.train(trainBatches);

    const TrainingHistory& history = trainer.getHistory();

    assert(earlyStop.shouldStopTraining());
    assert(earlyStop.hasBestMetric());
    assert(earlyStop.getBestEpoch() == 1);
    assert(earlyStop.getEpochsWithoutImprovement() == 2);

    // epoch 1 = improvement, epochs 2 and 3 = bad epochs, then stop
    assert(history.trainLosses.size() == 3);

    std::cout << "[PASS] EarlyStoppingCallback\n";
}

void testGenerationCallback() {
    CharTokenizer tokenizer;
    tokenizer.buildFromText("abc abc");

    Random modelRng(42);

    Transformer model(
        tokenizer.vocabSize(),
        4,  // context length
        4,  // embedding dimension
        8,  // FFN hidden dimension
        1,  // transformer block
        modelRng
    );

    SGDOptimizer optimizer(0.1f);
    TrainingHistory history;

    GenerationConfig generationConfig(
        3,      // maxNewTokens
        1.0f,   // temperature
        2,      // topK
        false,  // greedy sampling
        false,  // print generated text from config
        false,  // print token IDs
        123     // random seed
    );

    GenerationCallback callback(
        tokenizer,
        "a",
        generationConfig,
        2,      // generate every 2 epochs
        false,  // suppress callback output during test
        123
    );

    EpochContext context;
    context.totalEpochs = 4;
    context.globalStep = 1;
    context.trainLoss = 1.0f;
    context.hasValidationLoss = false;
    context.device = Device::CPU;
    context.runName = "generation_callback_test";

    // Epoch 1 should be skipped.
    context.epoch = 1;

    callback.onEpochEnd(
        context,
        model,
        optimizer,
        history
    );

    assert(callback.getSnapshots().empty());

    // Epoch 2 should create a snapshot.
    context.epoch = 2;

    callback.onEpochEnd(
        context,
        model,
        optimizer,
        history
    );

    assert(callback.getSnapshots().size() == 1);

    const GenerationSnapshot& firstSnapshot =
        callback.getLatestSnapshot();

    assert(firstSnapshot.epoch == 2);

    // Prompt length 1 + 3 generated characters.
    assert(firstSnapshot.text.size() == 4);
    assert(firstSnapshot.text[0] == 'a');

    // Epoch 3 should be skipped.
    context.epoch = 3;

    callback.onEpochEnd(
        context,
        model,
        optimizer,
        history
    );

    assert(callback.getSnapshots().size() == 1);

    // Epoch 4 should produce a second snapshot.
    context.epoch = 4;

    callback.onEpochEnd(
        context,
        model,
        optimizer,
        history
    );

    assert(callback.getSnapshots().size() == 2);
    assert(callback.getLatestSnapshot().epoch == 4);

    callback.clearSnapshots();
    assert(callback.getSnapshots().empty());

    std::cout << "[PASS] GenerationCallback\n";
}

void testCharTokenizerRoundTrip() {
    CharTokenizer tokenizer;

    const std::string text = "To be,\nOr not.";

    tokenizer.train(text);

    std::vector<int> ids = tokenizer.encode(text);
    std::string decoded = tokenizer.decode(ids);

    assert(!ids.empty());
    assert(decoded == text);
    assert(tokenizer.vocabSize() > 0);
    assert(tokenizer.isTrained());

    std::cout << "[PASS] Character Tokenizer Round Trip" << std::endl;
}

void testWordTokenizerRoundTrip() {
    WordTokenizer tokenizer(
        0,      // unlimited vocabulary
        1,      // minimum frequency
        true,   // preserve whitespace
        true    // preserve punctuation
    );

    const std::string text =
        "To be, or not to be.\nThat is the question.";

    tokenizer.train(text);

    std::vector<int> ids = tokenizer.encode(text);
    std::string decoded = tokenizer.decode(ids);

    assert(!ids.empty());
    assert(decoded == text);
    assert(tokenizer.isTrained());

    std::cout << "[PASS] Word Tokenizer Round Trip" << std::endl;
}

void testBPETokenizerRoundTrip() {
    BPETokenizer tokenizer(
        64,
        2
    );

    const std::string corpus =
        "low lower lowest low lower lowest "
        "newer wider lower";

    tokenizer.train(corpus);

    std::vector<int> ids =
        tokenizer.encode(corpus);

    std::string decoded =
        tokenizer.decode(ids);

    assert(!ids.empty());
    assert(decoded == corpus);
    assert(tokenizer.isTrained());
    assert(tokenizer.vocabSize() > 0);
    assert(!tokenizer.merges().empty());

    std::cout << "[PASS] BPE Tokenizer Round Trip" << std::endl;
}

void testBPETokenizerDeterminism() {
    const std::string corpus =
        "banana bandana banana bandana";

    BPETokenizer first(32, 2);
    BPETokenizer second(32, 2);

    first.train(corpus);
    second.train(corpus);

    assert(first.vocabSize() == second.vocabSize());
    assert(first.merges() == second.merges());

    std::vector<int> firstIds =
        first.encode(corpus);

    std::vector<int> secondIds =
        second.encode(corpus);

    assert(firstIds == secondIds);

    std::cout << "[PASS] BPE Tokenizer Determinism" << std::endl;
}

void testCharTokenizerSaveLoad() {
    const std::string path =
        "./test_char_tokenizer.tok";

    const std::string text =
        "abc ABC\n";

    CharTokenizer original;
    original.train(text);

    assert(original.save(path));

    CharTokenizer loaded;
    assert(loaded.load(path));

    assert(
        loaded.encode(text) ==
        original.encode(text)
    );

    assert(
        loaded.decode(
            loaded.encode(text)
        ) == text
    );

    std::filesystem::remove(path);

    std::cout << "[PASS] Character Tokenizer Save/Load" << std::endl;
}

void testBPETokenizerSaveLoad() {
    const std::string path =
        "./test_bpe_tokenizer.tok";

    const std::string corpus =
        "to be or not to be "
        "to be or not to be";

    BPETokenizer original(64, 2);
    original.train(corpus);

    assert(original.save(path));

    BPETokenizer loaded;
    assert(loaded.load(path));

    assert(
        loaded.encode(corpus) ==
        original.encode(corpus)
    );

    assert(
        loaded.decode(
            loaded.encode(corpus)
        ) == corpus
    );

    assert(
        loaded.merges() ==
        original.merges()
    );

    std::filesystem::remove(path);

    std::cout << "[PASS] BPE Tokenizer Save/Load" << std::endl;
}

void testTokenizerFactory() {
    TokenizerConfig config;

    config.type = TokenizerType::Character;
    std::unique_ptr<Tokenizer> character =
        createTokenizer(config);

    assert(character);
    assert(
        character->type() ==
        TokenizerType::Character
    );

    config.type = TokenizerType::Word;
    std::unique_ptr<Tokenizer> word =
        createTokenizer(config);

    assert(word);
    assert(
        word->type() ==
        TokenizerType::Word
    );

    config.type = TokenizerType::BPE;
    config.vocabSize = 64;
    config.minFrequency = 2;

    std::unique_ptr<Tokenizer> bpe =
        createTokenizer(config);

    assert(bpe);
    assert(
        bpe->type() ==
        TokenizerType::BPE
    );

    std::cout << "[PASS] Tokenizer Factory" << std::endl;
}

void testTextDatasetCharacterTokenizer() {
    TokenizerConfig config;
    config.type = TokenizerType::Character;

    TextDataset dataset(
        "./data/tiny_shakespeare.txt",
        32,
        config
    );

    assert(dataset.vocabSize() > 0);
    assert(dataset.numWindows() > 0);
    assert(dataset.tokenCount() > 32);

    assert(
        dataset.tokenizer().type() ==
        TokenizerType::Character
    );
    std::cout << "Dataset Character Tokenizer" << std::endl;
}

void testTextDatasetBPETokenizer() {
    const std::string tokenizerPath =
        "./test_tokenizers/test_bpe.tok";

    TokenizerConfig config;
    config.type = TokenizerType::BPE;
    config.vocabSize = 128;
    config.minFrequency = 2;
    config.modelPath = tokenizerPath;
    config.trainIfMissing = true;

    TextDataset dataset(
        "./data/tiny_shakespeare.txt",
        32,
        config
    );

    assert(dataset.vocabSize() > 0);
    assert(dataset.numWindows() > 0);

    assert(
        dataset.tokenizer().type() ==
        TokenizerType::BPE
    );

    assert(
        std::filesystem::exists(
            tokenizerPath
        )
    );

    std::filesystem::remove(
        tokenizerPath
    );

    std::cout << "[PASS] Dataset BPE Tokenizer" << std::endl;
}



/*
_______________________________________________________________________________________________________________________________________________________________
Add more test functions as needed
---------------------------------------------------------------------------------------------------------------------------------------------------------------
*/

int main(int argc, char** argv) {
    std::cout << "Transformer_Toy build OK\n" << std::endl;
	std::cout << "validating arguments..." << std::endl;
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    bool runSmokeTest = false;
	Device deviceSmoke = Device::AUTO;

	bool runShakedownTest = false;
	Device deviceShakedown = Device::AUTO;

	bool runProfilerTests = false;
    Device deviceProfiler = Device::AUTO;

    bool bypassBasicTests = false;
	bool runSpecificTests = false;
    bool runCoreTests = false;
	bool runAccelTests = false;
	bool runForwardParityTests = false;
	bool runBackwardTests = false;
	bool runBackwardParityTests = false;
	bool runOptimizerParityTests = false;
	bool runConfigTests = false;
    bool runCheckpointTests = false;
    bool runGenerationTests = false;
    bool runGenerationFineTune = false;
    bool runTokenizerTests = false;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--smoke") {
            runSmokeTest = true;
            std::cout << "Running smoke tests enabled." << std::endl;
            if (std::string(argv[i + 1]) == "-C" || std::string(argv[i + 1]) == "-c") {
				deviceSmoke = Device::CPU;
				std::cout << "Device set to CPU." << std::endl;
                i++; // Skip the next argument since it's the device flag
			}
			else if (std::string(argv[i + 1]) == "-G" || std::string(argv[i + 1]) == "-g") {
				deviceSmoke = Device::CUDA;
				std::cout << "Device set to CUDA." << std::endl;
				i++; // Skip the next argument since it's the device flag
			}
        }
		else if (std::string(argv[i]) == "--shakedown") {
			runShakedownTest = true;
			std::cout << "Running shakedown tests enabled." << std::endl;
			if (std::string(argv[i + 1]) == "-C" || std::string(argv[i + 1]) == "-c") {
				deviceShakedown = Device::CPU;
				std::cout << "Device set to CPU." << std::endl;
                i++; // Skip the next argument since it's the device flag
			}
			else if (std::string(argv[i + 1]) == "-G" || std::string(argv[i + 1]) == "-g") {
				deviceShakedown = Device::CUDA;
				std::cout << "Device set to CUDA." << std::endl;
                i++; // Skip the next argument since it's the device flag
			}
		}
        else if (std::string(argv[i]) == "--profile") {
            runProfilerTests = true;
            std::cout << "Running profiler tests enabled." << std::endl;
            if (std::string(argv[i + 1]) == "-C" || std::string(argv[i + 1]) == "-c") {
                deviceProfiler = Device::CPU;
                std::cout << "Device set to CPU." << std::endl;
                i++; // Skip the next argument since it's the device flag
            }
            else if (std::string(argv[i + 1]) == "-G" || std::string(argv[i + 1]) == "-g") {
                deviceProfiler = Device::CUDA;
                std::cout << "Device set to CUDA." << std::endl;
                i++; // Skip the next argument since it's the device flag
            }
        }
		else if (std::string(argv[i]) == "--bypass") {
			bypassBasicTests = true;
			std::cout << "Bypassing core tests enabled." << std::endl;
		}
		else if (std::string(argv[i]) == "--core") {
			runCoreTests = true;
			runSpecificTests = true;
			std::cout << "Running core tests enabled." << std::endl;
		}
		else if (std::string(argv[i]) == "--accel") {
			runAccelTests = true;
			runSpecificTests = true;
			std::cout << "Running accelerator tests enabled." << std::endl;
		}
		else if (std::string(argv[i]) == "--forward-parity") {
			runForwardParityTests = true;
			runSpecificTests = true;
			std::cout << "Running forward parity tests enabled." << std::endl;
		}
		else if (std::string(argv[i]) == "--backward") {
			runBackwardTests = true;
			runSpecificTests = true;
			std::cout << "Running backward tests enabled." << std::endl;
		}
		else if (std::string(argv[i]) == "--backward-parity") {
			runBackwardParityTests = true;
			runSpecificTests = true;
			std::cout << "Running backward parity tests enabled." << std::endl;
		}
		else if (std::string(argv[i]) == "--optimizer-parity") {
			runOptimizerParityTests = true;
			runSpecificTests = true;
			std::cout << "Running optimizer parity tests enabled." << std::endl;
		}
		else if (std::string(argv[i]) == "--config") {
			runConfigTests = true;
			runSpecificTests = true;
			std::cout << "Running configuration tests enabled." << std::endl;
		}
        else if (std::string(argv[i]) == "--checkpoint") {
            runCheckpointTests = true;
            runSpecificTests = true;
            std::cout << "Running checkpointing tests enabled." << std::endl;
        }
        else if (std::string(argv[i]) == "--generate-best") {
            runGenerationTests = true;
            std::cout << "Best-checkpoint generation tests enabled." << std::endl;
        }
        else if (std::string(argv[i]) == "--fine-tune") {
            runGenerationFineTune = true;
            std::cout << "Best Checkpoint generation fine tune enabled." << std::endl;
        }
        else if (std::string(argv[i]) == "--tokenizer") {
            runTokenizerTests = true;
            runSpecificTests = true;
            std::cout << "Tokenizer tests enabled." << std::endl;
        }
        else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --smoke                 Run smoke tests\n";
			std::cout << "  --smoke -C/-c           Run smoke tests on CPU\n";
			std::cout << "  --smoke -G/-g           Run smoke tests on GPU\n";
			std::cout << "  --shakedown             Run shakedown tests\n";
			std::cout << "  --shakedown -C/-c       Run shakedown tests on CPU\n";
			std::cout << "  --shakedown -G/-g       Run shakedown tests on GPU\n";
            std::cout << "  --profiler              Run profiler benchmarking tests\n";
            std::cout << "  --profiler -C/-c        Run profiler benchmarking tests on CPU\n";
            std::cout << "  --profiler -G/-g        Run profiler benchmarking tests on GPU\n";
            std::cout << "  --bypass                Bypass basic tests\n";
            std::cout << "  --core                  Run core tests\n";
            std::cout << "  --accel                 Run accelerator tests\n";
            std::cout << "  --forward-parity        Run forward parity tests\n";
            std::cout << "  --backward              Run backward tests\n";
            std::cout << "  --backward-parity       Run backward parity tests\n";
            std::cout << "  --optimizer-parity      Run optimizer parity tests\n";
			std::cout << "  --config                Run configuration tests\n";
            std::cout << "  --checkpoint            Run checkpointing tests\n";
            std::cout << "  --generate-best         Load best checkpoint and run generation suite\n";
            std::cout << "  --tokenizer             Run tokenizer specific tests from update\n";
        }
		else {
			std::cerr << "Unknown option: " << argv[i] << "\n";
			std::cerr << "Use --help or -h for usage information.\n";
			return 1;
		}
		// TODO: Add more command-line argument parsing as needed to control testing flow, such as selecting specific tests to run or setting verbosity levels.
    }

    if (!bypassBasicTests) {
        if (runCoreTests || !runSpecificTests) {
            std::cout << "\n===================================================\n";
            std::cout << "||           Basic Tests                         ||\n";
            std::cout << "===================================================\n\n";
            coreTest();
            layersTest();
            dataTest();
            mhaTest();
            testParameterBasics();
            testOptimizerZeroGrad();
            testSGDOptimizerBasicStep();
            testSGDOptimizerWeightDecay();
            testAdamOptimizerFirstStep();
            testAdamOptimizerMultipleStepsConstantGrad();
            testAdamOptimizerWeightDecay();
            testAdamOptimizerZeroGradInherited();
            testCrossEntropyLossPerfectConfidence();
            testCrossEntropyLossUniformLogits();
            testCrossEntropyLossBatchAverage();
            testTrainingHistory();
            testCheckpointSaveLoad();
            testTrainerBasicTrainingLoop();
        }
		if (runConfigTests) {
			configTest();
		}
		if (runAccelTests || !runSpecificTests) {
			std::cout << "\n===================================================\n";
			std::cout << "||          Accelerator (CUDA) Tests             ||\n";
			std::cout << "===================================================\n\n";
			testTensorToCUDAAndBack();
			testTensorCudaCopyConstructor();
			try {
				testTensorFillCUDA();
				testTensorScaleCUDA();
				testTensorAddCUDA();
			}
			catch (const std::exception& e) {
				std::cerr << "\n[CUDA TEST FAILURE]\n" << e.what() << "\n";
			}
		}
		if (runForwardParityTests || !runSpecificTests) {
            std::cout << "\n==================================================\n";
            std::cout << "||          Forward CUDA Parity Tests           ||\n";
            std::cout << "==================================================\n\n";
			testLinearForwardCUDAParity();
			testLinearClassForwardCUDAParity();
			testEmbeddingForwardCUDAParity();
            testLayerNormForwardCUDAParity();
			testCrossEntropyLossCUDAParity();
			testCrossEntropyLossCUDAUniformParity();
            testGeluCUDAParity();
            testSelfAttentionForwardCUDAParity();
			testMultiHeadAttentionForwardCUDAParity();
		}
        if (runOptimizerParityTests || !runSpecificTests) {
            std::cout << "\n==================================================\n";
            std::cout << "||          Optimizer CUDA Parity Tests         ||\n";
            std::cout << "==================================================\n\n";
            testSGDOptimizerCUDAParity();
            testAdamOptimizerCUDAParity();
            testAdamOptimizerCUDAParityMultipleSteps();
        }
        if (runBackwardTests || !runSpecificTests) {
            std::cout << "\n==================================================\n";
            std::cout << "||          Backwards function Tests            ||\n";
            std::cout << "==================================================\n\n";
            testLinearBackward();
            testFFNBackward();
            testLayerNormBackward();
            testSelfAttentionBackward();
            testTransformerBlockSingleHeadBackward();
            testMultiHeadAttentionBackward();
            testTransformerBlockMultiHeadBackward();
            testEmbeddingBackward();
            testTransformerBackwardSingleHead();
            testTransformerBackwardMultiHead();
        }
        if (runBackwardParityTests || !runSpecificTests) {
            std::cout << "\n==================================================\n";
            std::cout << "||          Backward CUDA Parity Tests          ||\n";
            std::cout << "==================================================\n\n";
            testLinearBackwardCUDAParity();
            testEmbeddingBackwardCUDAParity();
			testLayerNormBackwardCUDAParity();
            testGeluBackwardCUDAParity();
            testSelfAttentionBackwardCUDAParity();
            testMultiHeadAttentionBackwardCUDAParity();
        }
        if (runCheckpointTests || !runSpecificTests) {
            std::cout << "\n==================================================\n";
            std::cout << "||            Epoch Checkpoint Tests            ||\n";
            std::cout << "==================================================\n\n";
            testEpochCallback();
            testBestCheckpointCallback();
            testEarlyStoppingCallback();
            testGenerationCallback();
        }
        if (runTokenizerTests || !runSpecificTests) {
            std::cout << "\n==================================================\n";
            std::cout << "||            Tokenizer Update Tests            ||\n";
            std::cout << "==================================================\n\n";

            testCharTokenizerRoundTrip();
            testWordTokenizerRoundTrip();
            testBPETokenizerRoundTrip();
            testBPETokenizerDeterminism();
            testCharTokenizerSaveLoad();
            testBPETokenizerSaveLoad();
            testTokenizerFactory();
            testTextDatasetCharacterTokenizer();
            testTextDatasetBPETokenizer();
        }
    
		std::cout << "\n===================================================\n";
		std::cout << "||          All selected tests passed!           ||\n";
		std::cout << "===================================================\n\n";
    
    }

	if (runSmokeTest) {
        std::cout << "\n===================================================\n";
        std::cout << "||          Smoke Tests                          ||\n";
        std::cout << "===================================================\n\n";
		runSmokeTests(deviceSmoke);
	}

    if (runShakedownTest) {
        std::cout << "\n===================================================\n"
         << "||          Shakedown Tests                      ||\n"
         << "===================================================\n\n"
         << std::flush;
        
        runShakedownTests(deviceShakedown);
    }

    if (runGenerationTests) {
        std::cout << "\n==================================================\n"
            << "||          Generation Tests                   ||\n"
            << "=================================================\n" << std::flush;

        try {
            runBestModelGenerationTests();
        }
        catch (const std::exception& e) {
            std::cerr
                << "\n[BEST MODEL GENERATION FAILURE]\n"
                << e.what()
                << "\n";

            return 1;
        }
    }

    if (runGenerationFineTune) {
        std::cout << "\n==================================================\n"
            << "||          Generation Fine Tune                 ||\n"
            << "===================================================\n" << std::flush;
        runNarrowBandGenerationTests();
    }

    if (runProfilerTests) {
        runProfileBenchmark(deviceProfiler);
    }

	std::cout << "\n=================================================\n";
	std::cout << "||          All tests completed!                ||\n";
	std::cout << "=================================================\n\n";

    return 0;

}
