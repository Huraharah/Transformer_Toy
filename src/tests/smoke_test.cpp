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
#include "data/tokenizer.h"

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
            0.8f,
            5,
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
        config.logEverySteps = 0;

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
        config.generationEveryEpochs = 1;
        config.generationPrompt = "To be";

        config.generationConfig = GenerationConfig(
            40,     // maxNewTokens
            0.8f,   // temperature
            5,     // topK
            false,
            false,
            false,
            123
        );

        constexpr size_t contextLength = 64;
        constexpr size_t batchSize = 16;
        
        TokenizerConfig tokenizerChar = TokenizerConfig();
        tokenizerChar.modelPath.clear();
        tokenizerChar.trainIfMissing = true;
        TokenizerConfig tokenizerBPE = TokenizerConfig(TokenizerType::BPE, 128, 2);
        tokenizerBPE.modelPath = "./test_tokenizers/smoke_bpe_128.tok";
        tokenizerBPE.trainIfMissing = true;
        TokenizerConfig tokenizerWord = TokenizerConfig(TokenizerType::Word, 0, 1);
        tokenizerWord.preservePunctuation = true;
        tokenizerWord.preserveWhitespace = true;
        tokenizerWord.modelPath = "./test_tokenizers/smoke_word_full.tok";
        tokenizerWord.trainIfMissing = true;

        std::vector<TokenizerConfig> tokenizers = { tokenizerChar, tokenizerBPE, tokenizerWord};

        for (TokenizerConfig tokenizer : tokenizers) {
            TextDataset dataset("data/shakespeare.txt", contextLength, tokenizer);

            std::cout
                << "[INFO] Dataset loaded"
                << " | vocab=" << dataset.vocabSize()
                << " | context=" << dataset.contextLength()
                << " | windows=" << dataset.numWindows()
                << "\n";

            constexpr int dModel = 64;
            constexpr int dFF = 128;
            constexpr int numLayers = 2;

            config.checkpointDirectory += "_" + tokenizerTypeToString(tokenizer.type);
            config.runName += "_" + tokenizerTypeToString(tokenizer.type);

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
            blockConfig.attentionType = AttentionType::MultiHead;
            blockConfig.numHeads = 4;

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

            int numBatches;

            switch (tokenizer.type) {
            case TokenizerType::Character:
                numBatches = 20;
                break;

            case TokenizerType::BPE:
                numBatches = 50;
                break;

            case TokenizerType::Word:
                numBatches = 100;
                break;

            default:
                std::cerr << "Bad tokenizer type";
                numBatches = 0;
                break;
            }

            std::size_t activeBatchSize = batchSize;

            switch (tokenizer.type) {
            case TokenizerType::Character:
                activeBatchSize = 16;
                break;

            case TokenizerType::BPE:
                activeBatchSize = 16;
                break;

            case TokenizerType::Word:
                activeBatchSize = 8;
                break;
            }


            for (size_t i = 0; i < numBatches; ++i) {
                Tensor batchInputs;
                Tensor batchTargets;

                dataset.getBatch(activeBatchSize, rng, batchInputs, batchTargets);

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

            std::cout
                << "----------------[TOKENIZER]----------------\n"
                << "  type: " << tokenizer.type << "\n"
                << "  vocab size: " << dataset.vocabSize() << "\n"
                << "  raw characters: " << dataset.rawText().size() << "\n"
                << "  encoded tokens: " << dataset.tokenCount() << "\n"
                << "  compression ratio: "
                << static_cast<double>(dataset.rawText().size()) /
                static_cast<double>(dataset.tokenCount())
                << "\n----------------[Training]----------------\n";


            //std::cout << "[DEBUG] Constructing TrainingSession" << std::endl;

            TrainingSession session(
                model,
                loss,
                optimizer,
                config,
                &dataset.tokenizer()
            );

            //std::cout << "[DEBUG] TrainingSession Constructed, calling session.train()" << std::endl;

            session.train(
                trainBatches,
                &validationBatches
            );

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
                "_epoch_5.bin";   //TODO: Fix this to show the actual epoch, rather than statically epoch 5

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
                << "Vocab size: " << dataset.vocabSize() << "\n"
                << "Tokenizer: " << tokenizer.type << std::endl;
        }
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