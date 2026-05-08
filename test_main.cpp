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

#include <iostream>
#include <vector>

int main() {
    std::cout << "Transformer_Toy build OK\n" << std::endl;

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

    TransformerBlock block(4, 16, blockRng);

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

    Transformer model(
        10,  // vocabSize
        8,   // maxSequenceLength
        4,   // embedDim
        16,  // hiddenDim
        2,   // numLayers
        modelRng
    );

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

    Random genRng(987);

    Transformer genModel(
        dataset.vocabSize(),
        32,
        32,
        128,
        2,
        genRng
    );

    std::string prompt = "First Citizen:";

    Random sampleRng(555);

    std::string generated = genModel.generate(
        prompt,
        dataset.tokenizer(),
        80,
		2.0f, //temperature
        sampleRng
    );

    std::cout << "Prompt:\n" << prompt << "\n\n";
    std::cout << "Generated:\n" << generated << "\n\n";

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

    return 0;
}