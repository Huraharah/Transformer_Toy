#include "tests/best_model_generation_tests.h"

#include "core/random.h"
#include "data/dataset.h"
#include "layers/config.h"
#include "model/transformer.h"
#include "training/checkpoint.h"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    struct GenerationTestCase {
        std::string prompt;
        float temperature;
        size_t topK;
    };

    void printSection(const std::string& title) {
        std::cout
            << "\n============================================================\n"
            << "|| " << title << "\n"
            << "============================================================\n" << std::endl;
    }

    void moveModelToCPU(Transformer& model) {
        std::vector<Parameter*> parameters =
            model.parameters();

        for (Parameter* parameter : parameters) {
            if (!parameter) {
                continue;
            }

            parameter->value.toCPU();
            parameter->grad.toCPU();
        }
    }

} // namespace

void runBestModelGenerationTests() {
    printSection("Best Checkpoint Generation Tests");

    constexpr size_t contextLength = 64;

    constexpr int dModel = 64;
    constexpr int dFF = 128;
    constexpr int numLayers = 2;
    constexpr int numHeads = 4;

    constexpr size_t maxNewTokens = 500;
    constexpr uint32_t generationSeed = 2026;

    const std::string datasetPath =
        "data/shakespeare.txt";

    const std::string checkpointPath =
        "./shakedown_checkpoints/"
        "tiny_shakespeare_cuda_shakedown_best.bin";

    if (!std::filesystem::exists(checkpointPath)) {
        throw std::runtime_error(
            "Best checkpoint was not found: " +
            checkpointPath
        );
    }

    /*
        Rebuild the tokenizer from the same corpus.

        CharTokenizer uses an ordered character set, so rebuilding it from
        the same corpus reproduces the same character-to-token mapping.
    */
    TextDataset dataset(
        datasetPath,
        contextLength
    );

    /*
        Reconstruct the exact architecture used during training.
    */
    TransformerBlockConfig blockConfig;

    blockConfig.d_model = dModel;
    blockConfig.d_ff = dFF;
    blockConfig.pre_norm = true;
    blockConfig.use_bias = true;
    blockConfig.residual_dropout = 0.0f;
    blockConfig.ffn_dropout = 0.0f;
    blockConfig.attentionType =
        AttentionType::MultiHead;
    blockConfig.numHeads = numHeads;

    TransformerModelConfig modelConfig;

    modelConfig.vocab_size =
        static_cast<int>(dataset.vocabSize());

    modelConfig.max_seq_len =
        static_cast<int>(contextLength);

    modelConfig.num_layers = numLayers;
    modelConfig.block = blockConfig;
    modelConfig.learned_positional_embeddings = true;
    modelConfig.embedding_dropout = 0.0f;

    modelConfig.validate();

    /*
        These initial random values are temporary. Checkpoint::load()
        overwrites every model parameter.
    */
    Random modelRandom(42);

    Transformer model(
        modelConfig,
        modelRandom
    );

    std::vector<Parameter*> parameters =
        model.parameters();

    CheckpointMetadata metadata;
    TrainingHistory checkpointHistory;

    Checkpoint::load(
        checkpointPath,
        parameters,
        metadata,
        checkpointHistory
    );

    /*
        Checkpoint tensors are loaded as CPU tensors. Keep the inference
        model on CPU because the current autoregressive generation path
        reads sampled logits through host-side tensor access.
    */
    moveModelToCPU(model);

    std::cout
        << "[MODEL LOADED]\n"
        << "  checkpoint: " << checkpointPath << "\n"
        << "  best epoch: " << metadata.epoch << "\n"
        << "  global step: " << metadata.globalStep << "\n"
        << "  run name: " << metadata.runName << "\n"
        << "  vocabulary: " << dataset.vocabSize() << "\n"
        << "  context length: " << contextLength << "\n"
        << "  d_model: " << dModel << "\n"
        << "  d_ff: " << dFF << "\n"
        << "  layers: " << numLayers << "\n"
        << "  heads: " << numHeads << std::endl;

    /*
        You can either explicitly list cases or generate a full Cartesian
        product. The loop below generates every combination.
    */
    const std::vector<std::string> prompts = {
        "To be",
        "KING RICHARD:",
        "My lord,",
        "O, that",
        "ROMEO:"
    };

    const std::vector<float> temperatures = {
        0.5f,
        0.7f,
        0.9f,
        1.1f
    };

    const std::vector<size_t> topKValues = {
        3,
        5,
        10,
        20
    };

    size_t testNumber = 0;

    for (const std::string& prompt : prompts) {
        for (float temperature : temperatures) {
            for (size_t topK : topKValues) {
                ++testNumber;

                /*
                    Recreate the RNG for every generation.

                    Every test therefore begins with the identical random
                    state. Differences come from the prompt, temperature,
                    top-k restriction, and model probabilities—not from an
                    RNG that was advanced by earlier generations.
                */
                Random generationRandom(
                    generationSeed
                );

                std::string generatedText =
                    model.generate(
                        prompt,
                        dataset.tokenizer(),
                        maxNewTokens,
                        temperature,
                        topK,
                        generationRandom
                    );

                if (generatedText.empty()) {
                    throw std::runtime_error(
                        "Generation test produced empty output."
                    );
                }

                std::cout
                    << "\n------------------------------------------------------------\n"
                    << "[GENERATION TEST " << testNumber << "]\n"
                    << "Prompt: \"" << prompt << "\"\n"
                    << "Temperature: "
                    << std::fixed
                    << std::setprecision(2)
                    << temperature << "\n"
                    << "Top-K: " << topK << "\n"
                    << "Seed: " << generationSeed << "\n"
                    << "Generated tokens: "
                    << maxNewTokens << "\n"
                    << "------------------------------------------------------------\n"
                    << generatedText
                    << "\n------------------------------------------------------------" << std::endl;
            }
        }
    }

    std::cout
        << "\n[PASS] Best-checkpoint generation suite completed." <<std::endl;
}

void runNarrowBandGenerationTests() {
    printSection("Fine Tuning Generation Tests");

    constexpr size_t contextLength = 64;

    constexpr int dModel = 64;
    constexpr int dFF = 128;
    constexpr int numLayers = 2;
    constexpr int numHeads = 4;

    constexpr size_t maxNewTokens = 500;

    const std::string datasetPath =
        "data/shakespeare.txt";

    const std::string checkpointPath =
        "./shakedown_checkpoints/"
        "tiny_shakespeare_cuda_shakedown_best.bin";

    if (!std::filesystem::exists(checkpointPath)) {
        throw std::runtime_error(
            "Best checkpoint was not found: " +
            checkpointPath
        );
    }

    /*
        Rebuild the tokenizer from the same corpus.

        CharTokenizer uses an ordered character set, so rebuilding it from
        the same corpus reproduces the same character-to-token mapping.
    */
    TextDataset dataset(
        datasetPath,
        contextLength
    );

    /*
        Reconstruct the exact architecture used during training.
    */
    TransformerBlockConfig blockConfig;

    blockConfig.d_model = dModel;
    blockConfig.d_ff = dFF;
    blockConfig.pre_norm = true;
    blockConfig.use_bias = true;
    blockConfig.residual_dropout = 0.0f;
    blockConfig.ffn_dropout = 0.0f;
    blockConfig.attentionType =
        AttentionType::MultiHead;
    blockConfig.numHeads = numHeads;

    TransformerModelConfig modelConfig;

    modelConfig.vocab_size =
        static_cast<int>(dataset.vocabSize());

    modelConfig.max_seq_len =
        static_cast<int>(contextLength);

    modelConfig.num_layers = numLayers;
    modelConfig.block = blockConfig;
    modelConfig.learned_positional_embeddings = true;
    modelConfig.embedding_dropout = 0.0f;

    modelConfig.validate();

    /*
        These initial random values are temporary. Checkpoint::load()
        overwrites every model parameter.
    */
    Random modelRandom(42);

    Transformer model(
        modelConfig,
        modelRandom
    );

    std::vector<Parameter*> parameters =
        model.parameters();

    CheckpointMetadata metadata;
    TrainingHistory checkpointHistory;

    Checkpoint::load(
        checkpointPath,
        parameters,
        metadata,
        checkpointHistory
    );

    /*
        Checkpoint tensors are loaded as CPU tensors. Keep the inference
        model on CPU because the current autoregressive generation path
        reads sampled logits through host-side tensor access.
    */
    moveModelToCPU(model);

    std::cout
        << "[MODEL LOADED]\n"
        << "  checkpoint: " << checkpointPath << "\n"
        << "  best epoch: " << metadata.epoch << "\n"
        << "  global step: " << metadata.globalStep << "\n"
        << "  run name: " << metadata.runName << "\n"
        << "  vocabulary: " << dataset.vocabSize() << "\n"
        << "  context length: " << contextLength << "\n"
        << "  d_model: " << dModel << "\n"
        << "  d_ff: " << dFF << "\n"
        << "  layers: " << numLayers << "\n"
        << "  heads: " << numHeads << std::endl;

    /*
        You can either explicitly list cases or generate a full Cartesian
        product. The loop below generates every combination.
    */
    const std::vector<std::string> prompts = {
        "To be",
        "KING RICHARD:",
        "My lord,",
        "O, that",
        "ROMEO:"
    };

    const std::vector<float> temperatures = {
        0.80f,
        0.85f,
        0.90f,
        0.95f
    };

    const std::vector<size_t> topKValues = {
        4,
        5,
        6,
        7
    };

    const std::vector<int> randomSeeds = {
        42,
        1337,
        420,
        69,
        67
    };

    size_t testNumber = 0;

    for (const std::string& prompt : prompts) {
        for (int seed : randomSeeds) {
            for (float temperature : temperatures) {
                for (size_t topK : topKValues) {
                    ++testNumber;

                    /*
                        Recreate the RNG for every generation.

                        Each prompt/temperature/top-k combination is tested against the
                        same fixed set of seeds. This makes results directly comparable
                        across configurations while also measuring sampling variability.
                    */
                    Random generationRandom(
                        seed
                    );

                    std::string generatedText =
                        model.generate(
                            prompt,
                            dataset.tokenizer(),
                            maxNewTokens,
                            temperature,
                            topK,
                            generationRandom
                        );

                    if (generatedText.empty()) {
                        throw std::runtime_error(
                            "Generation test produced empty output."
                        );
                    }

                    std::cout
                        << "\n------------------------------------------------------------\n"
                        << "[GENERATION TEST " << testNumber << "]\n"
                        << "Prompt: \"" << prompt << "\"\n"
                        << "Temperature: "
                        << std::fixed
                        << std::setprecision(2)
                        << temperature << "\n"
                        << "Top-K: " << topK << "\n"
                        << "Seed: " << seed << "\n"
                        << "Generated tokens: "
                        << maxNewTokens << "\n"
                        << "------------------------------------------------------------\n"
                        << generatedText
                        << "\n------------------------------------------------------------" << std::endl;
                }
            }
        }
    }

    std::cout
        << "\n[PASS] Best-checkpoint generation suite completed." << std::endl;
}