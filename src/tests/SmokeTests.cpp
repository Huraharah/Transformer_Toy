#include "tests/SmokeTests.h"

#include "training/training_config.h"
#include "core/tensor.h"
#include "core/cuda_utils.h"
#include "data/dataset.h"
#include "core/random.h"
#include "training/trainer.h"
#include "layers/config.h"
#include "model/transformer.h"
#include "training/cross_entropy_loss.h"
#include "training/adam_optimizer.h"

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>

namespace {

    void printSection(const std::string& title) {
        std::cout << "\n==================================================\n";
        std::cout << "||               " << title << "                      ||\n";
        std::cout << "==================================================\n\n";
    }

    bool cudaAvailableForSmokeTest() {
        return isCudaAvailable();
    }

    std::string runGenerationSmokeCheck(
        Transformer& model,
        const TextDataset& dataset,
        Random& rng
    ) {
        std::string generated = model.generate(
            "To be",
            dataset.tokenizer(),
            100,
            1.0f,
            10,
            rng
        );

        if (generated.empty()) {
            throw std::runtime_error("Generation smoke check produced empty output.");
        }

		return generated;
    }

    void runTinyShakespeareSmokeTest(Device device) {
        std::cout << "[SMOKE] Tiny Shakespeare test on "
            << (device == Device::CUDA ? "CUDA" : "CPU")
            << "\n";

        TrainingConfig config;
        config.device = device;
        config.epochs = 10;
        config.batchSize = 16;
        config.logEverySteps = 20;
        config.checkpointEveryEpochs = 0;
        config.runName = device == Device::CUDA
            ? "tiny_shakespeare_cuda_smoke"
            : "tiny_shakespeare_cpu_smoke";
        config.checkpointDirectory = ".";

        constexpr size_t contextLength = 64;
        constexpr size_t batchSize = 16;

        TextDataset dataset("data/shakespeare.txt", contextLength);

        std::cout
            << "[INFO] Dataset loaded"
            << " | vocab=" << dataset.vocabSize()
            << " | context=" << dataset.contextLength()
            << " | windows=" << dataset.numWindows()
            << "\n";


        constexpr int dModel = 64;
        constexpr int dFF = 32;
        constexpr int numLayers = 2;

        TransformerBlockConfig blockConfig;
        blockConfig.d_model = dModel;
        blockConfig.d_ff = dFF;
        blockConfig.pre_norm = true;
        blockConfig.use_bias = true;
        blockConfig.residual_dropout = 0.0f;
        blockConfig.ffn_dropout = 0.0f;
        blockConfig.attentionType = AttentionType::SingleHead;
        blockConfig.numHeads = 1;

        TransformerModelConfig modelConfig;
        modelConfig.vocab_size = static_cast<int>(dataset.vocabSize());
        modelConfig.max_seq_len = static_cast<int>(dataset.contextLength());
        modelConfig.num_layers = numLayers;
        modelConfig.block = blockConfig;
        modelConfig.learned_positional_embeddings = true;
        modelConfig.embedding_dropout = 0.0f;

        modelConfig.validate();

        Random rng(42);

        Transformer model(modelConfig, rng);

        Tensor inputs;
        Tensor targets;

        dataset.getBatch(batchSize, rng, inputs, targets);

        std::cout
            << "[INFO] Batch created"
            << " | inputs=" << inputs.shapeString()
            << " | targets=" << targets.shapeString()
            << "\n";

        TrainingBatch batch;
        batch.inputs = inputs;
        batch.targets = targets;

        std::vector<TrainingBatch> trainBatches;


        for (size_t i = 0; i < 20; ++i) {
            Tensor batchInputs;
            Tensor batchTargets;

            dataset.getBatch(batchSize, rng, batchInputs, batchTargets);

            trainBatches.push_back({ batchInputs, batchTargets });
        }

        CrossEntropyLoss loss;
        AdamOptimizer optimizer(0.001f);

        Trainer trainer(model, loss, optimizer, config);
        trainer.train(trainBatches);

        std::cout << "[PASS] Smoke test scaffold completed for "
            << (device == Device::CUDA ? "CUDA" : "CPU")
            << "\n";

        const TrainingHistory& history = trainer.getHistory();

        if (history.trainLosses.empty()) {
            throw std::runtime_error("Smoke test produced no training losses.");
        }

        for (float value : history.trainLosses) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("Smoke test produced non-finite loss.");
            }
        }

        float improvement =
            100.0f *
            (
                history.trainLosses.front() -
                history.trainLosses.back()
                ) /
            history.trainLosses.front();

        std::cout
            << "====================================================\n"
            << "||            Model Training Summary              ||\n"
            << "====================================================\n"
            << "Initial loss = " << history.trainLosses.front()
            << "\nFinal loss = " << history.trainLosses.back() << "\n";
        std::cout
            << std::fixed
            << std::setprecision(2)
            << "Improvement = "
            << improvement
            << "%\n";
        std::cout
			<< "====================================================\n"
			<< "||            Model Generation Sample             ||\n"
			<< "====================================================\n"
			<< "Prompt: \"To be\"\n"
			<< "---------------------------------------------------\n"
			<< "Generated sample:\n";

        std::string generatedSample = runGenerationSmokeCheck(model, dataset, rng);

		std::cout << generatedSample << "\n"
			<< "---------------------------------------------------\n";

        std::cout << std::setprecision(6);

        std::cout
            << "==================================================\n"
            << "||            Model Configuration               ||\n"
            << "==================================================\n"
            << "Device: " << (device == Device::CUDA ? "CUDA" : "CPU") << "\n"
            << "Layers: " << numLayers << "\n"
            << "Embedding dim: " << dModel << "\n"
            << "Feedforward dim: " << dFF << "\n"
            << "Heads: " << blockConfig.numHeads << "\n"
            << "Context: " << dataset.contextLength() << "\n"
            << "Batch size: " << batchSize << "\n"
            << "Epochs: " << config.epochs << "\n"
            << "Learning Rate: " << optimizer.getLearningRate() << "\n"
            << "Vocab size: " << dataset.vocabSize() << "\n";
    }

}

void runSmokeTests(Device device) {
    printSection("Smoke Tests");

    if (device == Device::CPU || device == Device::AUTO) {
        runTinyShakespeareSmokeTest(Device::CPU);
    }
    if (device == Device::CUDA || device == Device::AUTO) {
		if (cudaAvailableForSmokeTest()) {
			runTinyShakespeareSmokeTest(Device::CUDA);
		}
		else {
			std::cout << "[SKIP] CUDA not available, skipping CUDA smoke test\n";
		}
    }

    std::cout << "\n[PASS] Smoke test suite completed\n";
}