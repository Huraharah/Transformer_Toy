#include "tests/smoke_test.h"
#include "training/training_config.h"
#include "core/tensor.h"
#include "core/cuda_utils.h"
#include "data/dataset.h"
#include "core/random.h"
#include "training/training_session.h"
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
#include <filesystem>

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

        config.runName = device == Device::CUDA
            ? "tiny_shakespeare_cuda_smoke"
            : "tiny_shakespeare_cpu_smoke";

        config.checkpointDirectory = "./smoke_checkpoints";

        // Periodic checkpoint
        config.enableCheckpointing = true;
        config.checkpointEveryEpochs = 5;

        // Best checkpoint
        config.enableBestCheckpoint = true;
        config.bestCheckpointMinDelta = 0.0f;

        // Validation metric
        config.preferValidationLoss = true;

        // Early stopping
        config.enableEarlyStopping = true;
        config.earlyStoppingPatience = 20;
        config.earlyStoppingMinDelta = 0.0f;

        // Scheduler
        config.enableLRScheduler = true;
        config.lrSchedule = LearningRateSchedule::StepDecay;
        config.lrStepSize = 5;
        config.lrGamma = 0.5f;
        config.minimumLearningRate = 1.0e-6f;

        // Generation snapshots
        config.enableGenerationSnapshots = true;
        config.generationEveryEpochs = 2;
        config.generationPrompt = "To be";

        config.generationConfig = GenerationConfig(
            40,     // maxNewTokens
            1.0f,   // temperature
            10,     // topK
            false,
            false,
            false,
            123
        );

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

        std::filesystem::create_directories(
            config.checkpointDirectory
        );

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

        std::vector<TrainingBatch> validationBatches;

        for (size_t i = 0; i < 4; ++i) {
            Tensor batchInputs;
            Tensor batchTargets;

            dataset.getBatch(
                batchSize,
                rng,
                batchInputs,
                batchTargets
            );

            validationBatches.push_back({
                batchInputs,
                batchTargets
                });
        }

        CrossEntropyLoss loss;
        AdamOptimizer optimizer(0.001f);

        float initialLearningRate =
            optimizer.getLearningRate();

        std::cout << "[PASS] Smoke test scaffold completed for "
            << (device == Device::CUDA ? "CUDA" : "CPU")
            << "\n";

        std::cout << "[DEBUG] Constructing session..." << std::endl;

        TrainingSession session(
            model,
            loss,
            optimizer,
            config,
            &dataset.tokenizer()
        );

        std::cout << "[DEBUG] TrainingSession constructed." << std::endl;
        std::cout << "[DEBUG] Entering session.train()..." << std::endl;

        session.train(
            trainBatches,
            &validationBatches
        );

        std::cout << "[DEBUG] session.train() returned." << std::endl;

        const TrainingHistory& history =
            session.getHistory();

        const size_t expectedMaximumTrainLosses =
            static_cast<size_t>(config.epochs) *
            trainBatches.size();

        if (history.trainLosses.empty()) {
            throw std::runtime_error(
                "Smoke test produced no training losses."
            );
        }

        if (history.trainLosses.size() > expectedMaximumTrainLosses) {
            throw std::runtime_error(
                "Smoke test produced too many training-history entries."
            );
        }

        if (history.validationLosses.empty()) {
            throw std::runtime_error(
                "Smoke test produced no validation losses."
            );
        }

        for (float value : history.trainLosses) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Smoke test produced non-finite training loss."
                );
            }
        }

        for (float value : history.validationLosses) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Smoke test produced non-finite validation loss."
                );
            }
        }

        const std::string bestCheckpointPath =
            config.checkpointDirectory + "/" +
            config.runName +
            "_best.bin";

        if (!std::filesystem::exists(bestCheckpointPath)) {
            throw std::runtime_error(
                "Smoke test did not create a best checkpoint."
            );
        }

        const std::string periodicCheckpointPath =
            config.checkpointDirectory + "/" +
            config.runName +
            "_epoch_5.bin";

        if (!std::filesystem::exists(periodicCheckpointPath)) {
            throw std::runtime_error(
                "Smoke test did not create the expected periodic checkpoint."
            );
        }

        float finalLearningRate =
            optimizer.getLearningRate();

        if (!(finalLearningRate < initialLearningRate)) {
            throw std::runtime_error(
                "Learning-rate scheduler did not reduce the learning rate."
            );
        }

        const GenerationCallback* generationCallback =
            session.getGenerationCallback();

        if (!generationCallback) {
            throw std::runtime_error(
                "Generation callback was not created."
            );
        }

        const auto& snapshots =
            generationCallback->getSnapshots();

        if (snapshots.empty()) {
            throw std::runtime_error(
                "Generation callback produced no snapshots."
            );
        }

        for (const GenerationSnapshot& snapshot : snapshots) {
            if (snapshot.text.empty()) {
                throw std::runtime_error(
                    "Generation callback produced an empty snapshot."
                );
            }
        }

        const GenerationSnapshot& latest =
            generationCallback->getLatestSnapshot();

        std::cout
            << "====================================================\n"
            << "||            Latest Generation Snapshot          ||\n"
            << "====================================================\n"
            << "Epoch: " << latest.epoch << "\n"
            << latest.text << "\n"
            << "---------------------------------------------------\n";

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

        std::vector<Parameter*> modelParameters =
            model.parameters();

        if (device == Device::CUDA) {
            for (Parameter* parameter : modelParameters) {
                if (!parameter) {
                    continue;
                }

                parameter->value.toCPU();
                parameter->grad.toCPU();
            }
        }

        std::string generatedSample =
            runGenerationSmokeCheck(model, dataset, rng);

        std::cout
            << generatedSample
            << "\n"
            << "---------------------------------------------------\n";

        if (device == Device::CUDA) {
            for (Parameter* parameter : modelParameters) {
                if (!parameter) {
                    continue;
                }

                parameter->value.toCUDA();
                parameter->grad.toCUDA();
            }
        }

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