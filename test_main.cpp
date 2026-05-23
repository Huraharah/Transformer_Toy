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

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

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

    std::vector<size_t> encoded = tokenizer.encode(text);
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

    TextDataset dataset("shakespeare.txt", 16);

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

    std::cout << "Input text:  [" << dataset.tokenizer().decode(inputIds) << "]\n";
    std::cout << "Target text: [" << dataset.tokenizer().decode(targetIds) << "]\n\n";

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

        std::cout << "Sample " << b << " input:  ["
            << dataset.tokenizer().decode(inputIds) << "]\n";

        std::cout << "Sample " << b << " target: ["
            << dataset.tokenizer().decode(targetIds) << "]\n";
    }

    std::cout << "\n==================================================\n";
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
    Tensor lossTargets;

    Random lossBatchRng(2468);
    dataset.getBatch(4, lossBatchRng, lossInputs, lossTargets);

    Tensor lossLogits = lossModel.forward(lossInputs);

    float loss = MathUtils::crossEntropyLoss(lossLogits, lossTargets);
    float ppl = MathUtils::perplexity(loss);
    float acc = MathUtils::tokenAccuracy(lossLogits, lossTargets);

    std::cout << "Input shape: " << lossInputs.shapeString() << "\n";
    std::cout << "Target shape: " << lossTargets.shapeString() << "\n";
    std::cout << "Logits shape: " << lossLogits.shapeString() << "\n";
    std::cout << "Cross-entropy loss: " << loss << "\n";
    std::cout << "Perplexity: " << ppl << "\n";
    std::cout << "Token accuracy: " << acc << "\n";

    std::cout << "\n==================================================\n";
    std::cout << "||            All core tests completed!         ||\n";
    std::cout << "==================================================\n";
}

void configTest() {
    std::cout << "\n~~~~~~~~~~~~~ CONFIG TESTS ~~~~~~~~~~~~~\n";

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

int main() {
    std::cout << "Transformer_Toy build OK\n" << std::endl;
    coreTest();
    layersTest();
    dataTest();
    configTest();
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

    return 0;

}
