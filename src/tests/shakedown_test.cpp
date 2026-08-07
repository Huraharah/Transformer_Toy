#include "tests/shakedown_test.h"
#include "core/cuda_utils.h"
#include "core/random.h"
#include "data/dataset.h"
#include "layers/config.h"
#include "model/transformer.h"
#include "training/adam_optimizer.h"
#include "training/checkpoint.h"
#include "training/cross_entropy_loss.h"
#include "training/training_config.h"
#include "training/training_session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    struct BatchBuildResult {
        std::vector<TrainingBatch> batches;

        size_t sequenceCount = 0;
        size_t inputTokenCount = 0;
        size_t transitionCount = 0;
        size_t tailSequenceLength = 0;
    };

    void printSection(const std::string& title) {
        std::cout
            << "\n============================================================\n"
            << "|| " << title << "\n"
            << "============================================================\n" << std::endl;
    }

    const char* deviceName(Device device) {
        switch (device) {
        case Device::CPU:
            return "CPU";

        case Device::CUDA:
            return "CUDA";

        case Device::AUTO:
            return "AUTO";

        default:
            return "UNKNOWN";
        }
    }

    /*
        Builds deterministic, contiguous batches over [rangeBegin, rangeEnd).

        Each sample is:

            input:  tokens[start ... start + sequenceLength - 1]
            target: tokens[start + 1 ... start + sequenceLength]

        Full context-length sequences are grouped into normal batches.

        If the range has a final remainder shorter than contextLength,
        one final single-sample batch is created with the shorter sequence.
        This means the corpus tail is not silently discarded.
    */
    BatchBuildResult buildSequentialBatches(
        const std::vector<int>& tokenIds,
        size_t rangeBegin,
        size_t rangeEnd,
        size_t contextLength,
        size_t batchSize
    ) {
        if (contextLength == 0) {
            throw std::invalid_argument(
                "Shakedown contextLength must be greater than zero."
            );
        }

        if (batchSize == 0) {
            throw std::invalid_argument(
                "Shakedown batchSize must be greater than zero."
            );
        }

        if (rangeBegin >= rangeEnd) {
            throw std::invalid_argument(
                "Sequential batch range must be non-empty."
            );
        }

        if (rangeEnd > tokenIds.size()) {
            throw std::out_of_range(
                "Sequential batch range exceeds token count."
            );
        }

        const size_t rangeTokenCount =
            rangeEnd - rangeBegin;

        if (rangeTokenCount < 2) {
            throw std::invalid_argument(
                "Sequential batch range must contain at least two tokens."
            );
        }

        BatchBuildResult result;

        /*
            One input/target transition requires two adjacent tokens.

            With N tokens in the range, there are N - 1 usable transitions.
        */
        const size_t usableTransitions =
            rangeTokenCount - 1;

        const size_t fullSequenceCount =
            usableTransitions / contextLength;

        const size_t tailLength =
            usableTransitions % contextLength;

        size_t sequenceIndex = 0;

        while (sequenceIndex < fullSequenceCount) {
            const size_t sequencesInBatch =
                std::min(
                    batchSize,
                    fullSequenceCount - sequenceIndex
                );

            Tensor inputs(
                { sequencesInBatch, contextLength },
                0.0f
            );

            Tensor targets(
                { sequencesInBatch, contextLength },
                0.0f
            );

            for (size_t b = 0; b < sequencesInBatch; ++b) {
                const size_t sequenceStart =
                    rangeBegin +
                    (sequenceIndex + b) * contextLength;

                for (size_t t = 0; t < contextLength; ++t) {
                    inputs.at({ b, t }) =
                        static_cast<float>(
                            tokenIds[sequenceStart + t]
                            );

                    targets.at({ b, t }) =
                        static_cast<float>(
                            tokenIds[sequenceStart + t + 1]
                            );
                }
            }

            result.batches.push_back({
                std::move(inputs),
                std::move(targets)
                });

            result.sequenceCount += sequencesInBatch;

            result.inputTokenCount +=
                sequencesInBatch * contextLength;

            result.transitionCount +=
                sequencesInBatch * contextLength;

            sequenceIndex += sequencesInBatch;
        }

        /*
            Preserve the final corpus tail instead of dropping it.

            This batch can have a sequence dimension smaller than the normal
            context length, which the Transformer already supports as long as
            the sequence length does not exceed max_seq_len.
        */
        if (tailLength > 0) {
            const size_t tailStart =
                rangeBegin +
                fullSequenceCount * contextLength;

            Tensor tailInputs(
                { 1, tailLength },
                0.0f
            );

            Tensor tailTargets(
                { 1, tailLength },
                0.0f
            );

            for (size_t t = 0; t < tailLength; ++t) {
                tailInputs.at({ 0, t }) =
                    static_cast<float>(
                        tokenIds[tailStart + t]
                        );

                tailTargets.at({ 0, t }) =
                    static_cast<float>(
                        tokenIds[tailStart + t + 1]
                        );
            }

            result.batches.push_back({
                std::move(tailInputs),
                std::move(tailTargets)
                });

            ++result.sequenceCount;
            result.inputTokenCount += tailLength;
            result.transitionCount += tailLength;
            result.tailSequenceLength = tailLength;
        }

        if (result.batches.empty()) {
            throw std::runtime_error(
                "Sequential batch builder produced no batches."
            );
        }

        if (result.transitionCount != usableTransitions) {
            throw std::runtime_error(
                "Sequential batch builder did not cover all corpus transitions."
            );
        }

        return result;
    }

    void validateFiniteHistory(
        const TrainingHistory& history
    ) {
        if (history.trainLosses.empty()) {
            throw std::runtime_error(
                "Shakedown produced no training losses."
            );
        }

        if (history.validationLosses.empty()) {
            throw std::runtime_error(
                "Shakedown produced no validation losses."
            );
        }

        for (float loss : history.trainLosses) {
            if (!std::isfinite(loss)) {
                throw std::runtime_error(
                    "Shakedown produced a non-finite training loss."
                );
            }
        }

        for (float loss : history.validationLosses) {
            if (!std::isfinite(loss)) {
                throw std::runtime_error(
                    "Shakedown produced a non-finite validation loss."
                );
            }
        }
    }

    void moveModelParametersToCPU(
        Transformer& model
    ) {
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

    void removeExistingRunFiles(
        const TrainingConfig& config
    ) {
        const std::filesystem::path directory =
            config.checkpointDirectory;

        if (!std::filesystem::exists(directory)) {
            return;
        }

        const std::string prefix =
            config.runName + "_";

        for (const auto& entry :
            std::filesystem::directory_iterator(directory)) {

            if (!entry.is_regular_file()) {
                continue;
            }

            const std::string filename =
                entry.path().filename().string();

            if (filename.rfind(prefix, 0) == 0) {
                std::filesystem::remove(entry.path());
            }
        }
    }

    void runFullDatasetShakedown(Device requestedDevice) {
        Device device =
            resolveDevice(requestedDevice);

        if (
            requestedDevice == Device::CUDA &&
            device != Device::CUDA
            ) {
            throw std::runtime_error(
                "CUDA shakedown was requested, but CUDA is unavailable."
            );
        }

        printSection(
            std::string("Full Dataset Shakedown - ") +
            deviceName(device)
        );

        /*
            Initial full-scale configuration.

            These values are intentionally substantial enough to prove the
            system can learn from the complete corpus, while still remaining
            practical for the first shakedown run.
        */
        constexpr size_t contextLength = 64;
        constexpr size_t batchSize = 16;

        constexpr int dModel = 64;
        constexpr int dFF = 128;
        constexpr int numLayers = 2;
        constexpr int numHeads = 4;

        constexpr int maximumEpochs = 25;

        constexpr float trainingFraction = 0.90f;

        TokenizerConfig tokenizer;
        tokenizer.type = TokenizerType::BPE;
        tokenizer.vocabSize = 512;
        tokenizer.minFrequency = 2;
        tokenizer.modelPath = "./tokenizers/complete_shakespeare_bpe_512.tok";
        tokenizer.trainIfMissing = true;

        std::filesystem::create_directories("./tokenizers");

        TextDataset dataset(
            "data/shakespeare_complete_cleaned.txt",
            contextLength,
            tokenizer
        );

        const std::vector<int> allTokenIds =
            dataset.tokenizer().encode(
                dataset.rawText()
            );

        if (allTokenIds.size() < 100) {
            throw std::runtime_error(
                "Dataset is unexpectedly small for the shakedown."
            );
        }

        /*
            Ensure each range has at least one target transition.

            The validation split begins at splitIndex. It is intentionally
            independent rather than sharing the train range's final token.
        */
        size_t splitIndex =
            static_cast<size_t>(
                static_cast<double>(allTokenIds.size()) *
                trainingFraction
                );

        splitIndex = std::clamp(
            splitIndex,
            static_cast<size_t>(2),
            allTokenIds.size() - 2
        );

        BatchBuildResult trainingBuild =
            buildSequentialBatches(
                allTokenIds,
                0,
                splitIndex,
                contextLength,
                batchSize
            );

        BatchBuildResult validationBuild =
            buildSequentialBatches(
                allTokenIds,
                splitIndex,
                allTokenIds.size(),
                contextLength,
                batchSize
            );

        std::cout
            << "[DATASET]\n"
            << "  raw characters: "
            << dataset.rawText().size() << "\n"
            << "  encoded tokens: "
            << allTokenIds.size() << "\n"
            << "  vocabulary: "
            << dataset.vocabSize() << "\n"
            << "  compression ratio: "
            << static_cast<double>(dataset.rawText().size()) /
            static_cast<double>(allTokenIds.size())
            << "\n"
            << "  training tokens: "
            << splitIndex << "\n"
            << "  training batches: "
            << trainingBuild.batches.size() << "\n"
            << "  validation batches: "
            << validationBuild.batches.size() << "\n"
            << "  training sequences: "
            << trainingBuild.sequenceCount << "\n"
            << "  validation sequences: "
            << validationBuild.sequenceCount << "\n"
            << "  train transitions covered: "
            << trainingBuild.transitionCount << "\n"
            << "  validation transitions covered: "
            << validationBuild.transitionCount << "\n"
            << "  train tail length: "
            << trainingBuild.tailSequenceLength << "\n"
            << "  validation tail length: "
            << validationBuild.tailSequenceLength << "\n"
            << "  tokenizer: BPE\n"
            << "  requested BPE vocabulary: "
            << tokenizer.vocabSize << "\n"
            << "  actual vocabulary: "
            << dataset.vocabSize() << "\n"
            << "  raw characters: "
            << dataset.rawText().size() << "\n"
            << "  encoded tokens: "
            << allTokenIds.size() << "\n"
            << "  compression ratio: "
            << static_cast<double>(dataset.rawText().size()) /
            static_cast<double>(allTokenIds.size())
            << std::endl;

        TrainingConfig config;

        config.device = device;
        config.epochs = maximumEpochs;
        config.batchSize =
            static_cast<int>(batchSize);

        config.logEverySteps = 100;
        //config.useValidation = true;
        //config.shuffle = false;

        config.runName =
            device == Device::CUDA
            ? "full_shakespeare_cuda_bpe_512_shakedown"
            : "full_shakespeare_cpu_bpe_512_shakedown";

        config.checkpointDirectory =
            "./shakedown_checkpoints_complete_bpe_512";

        // Periodic checkpoints
        config.enableCheckpointing = true;
        config.checkpointEveryEpochs = 1;

        // Best checkpoint
        config.enableBestCheckpoint = true;
        config.bestCheckpointMinDelta = 0.0001f;

        // Validation is the primary control metric.
        config.preferValidationLoss = true;

        // Early stopping
        config.enableEarlyStopping = true;
        config.earlyStoppingPatience = 5;
        config.earlyStoppingMinDelta = 0.0001f;

        // Learning-rate scheduler
        config.enableLRScheduler = true;
        config.lrSchedule =
            LearningRateSchedule::CosineDecay;

        config.lrStepSize = 1;
        config.lrGamma = 1.0f;
        config.minimumLearningRate = 1.0e-5f;

        // Generation snapshots
        config.enableGenerationSnapshots = true;
        config.generationEveryEpochs = 5;
        config.generationPrompt = "To be";

        config.generationConfig = GenerationConfig(
            80,     // maxNewTokens
            0.9f,   // temperature
            10,     // topK
            false,
            true,
            false,
            2026
        );

        std::filesystem::create_directories(
            config.checkpointDirectory
        );

        removeExistingRunFiles(config);

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

        Random modelRandom(42);

        Transformer model(
            modelConfig,
            modelRandom
        );

        CrossEntropyLoss loss;

        AdamOptimizer optimizer(
            0.001f,
            0.9f,
            0.999f,
            1.0e-8f,
            0.0f
        );

        const float initialLearningRate =
            optimizer.getLearningRate();

        TrainingSession session(
            model,
            loss,
            optimizer,
            config,
            &dataset.tokenizer()
        );

        std::cout
            << "\n[TRAINING]\n"
            << "  device: "
            << deviceName(device) << "\n"
            << "  maximum epochs: "
            << config.epochs << "\n"
            << "  context length: "
            << contextLength << "\n"
            << "  batch size: "
            << batchSize << "\n"
            << "  d_model: "
            << dModel << "\n"
            << "  d_ff: "
            << dFF << "\n"
            << "  layers: "
            << numLayers << "\n"
            << "  heads: "
            << numHeads << "\n"
            << "  initial learning rate: "
            << initialLearningRate << "\n" << std::endl;

        const auto trainingStart =
            std::chrono::steady_clock::now();

        session.train(
            trainingBuild.batches,
            &validationBuild.batches
        );

        const auto trainingEnd =
            std::chrono::steady_clock::now();

        const double elapsedSeconds =
            std::chrono::duration<double>(
                trainingEnd - trainingStart
            ).count();

        const TrainingHistory& history =
            session.getHistory();

        validateFiniteHistory(history);

        const size_t completedEpochs =
            history.validationLosses.size();

        if (completedEpochs == 0) {
            throw std::runtime_error(
                "Shakedown completed zero epochs."
            );
        }

        const size_t expectedTrainingSteps =
            completedEpochs *
            trainingBuild.batches.size();

        if (
            history.trainLosses.size() !=
            expectedTrainingSteps
            ) {
            throw std::runtime_error(
                "Training-history size does not match completed epochs."
            );
        }

        const std::string bestCheckpointPath =
            config.checkpointDirectory + "/" +
            config.runName +
            "_best.bin";

        if (
            !std::filesystem::exists(
                bestCheckpointPath
            )
            ) {
            throw std::runtime_error(
                "Shakedown did not create a best checkpoint."
            );
        }

        const GenerationCallback* generationCallback =
            session.getGenerationCallback();

        if (!generationCallback) {
            throw std::runtime_error(
                "Shakedown generation callback was not created."
            );
        }

        const auto& snapshots =
            generationCallback->getSnapshots();

        if (snapshots.empty()) {
            throw std::runtime_error(
                "Shakedown produced no generation snapshots."
            );
        }

        const size_t expectedSnapshots =
            completedEpochs /
            static_cast<size_t>(
                config.generationEveryEpochs
                );

        if (snapshots.size() != expectedSnapshots) {
            throw std::runtime_error(
                "Shakedown produced an unexpected number of generation snapshots."
            );
        }

        /*
            Restore the best model before final qualitative generation.

            Optimizer state is intentionally not restored in this milestone.
            Training has already finished, so only model state matters here.
        */
        std::vector<Parameter*> parameters =
            model.parameters();

        CheckpointMetadata bestMetadata;
        TrainingHistory bestCheckpointHistory;

        Checkpoint::load(
            bestCheckpointPath,
            parameters,
            bestMetadata,
            bestCheckpointHistory
        );

        moveModelParametersToCPU(model);

        Random generationRandom(9876);

        const std::string finalGeneratedText =
            model.generate(
                "To be",
                dataset.tokenizer(),
                500,
                0.8f,
                10,
                generationRandom
            );

        if (finalGeneratedText.empty()) {
            throw std::runtime_error(
                "Final best-checkpoint generation was empty."
            );
        }

        const float initialLoss =
            history.trainLosses.front();

        const float finalLoss =
            history.trainLosses.back();

        const float initialValidationLoss =
            history.validationLosses.front();

        const float finalValidationLoss =
            history.validationLosses.back();

        const float trainingImprovement =
            100.0f *
            (initialLoss - finalLoss) /
            initialLoss;

        const float validationImprovement =
            100.0f *
            (
                initialValidationLoss -
                finalValidationLoss
                ) /
            initialValidationLoss;

        const float finalLearningRate =
            optimizer.getLearningRate();

        printSection("Shakedown Results");

        std::cout
            << std::fixed
            << std::setprecision(6)
            << "Device: "
            << deviceName(device) << "\n"
            << "Completed epochs: "
            << completedEpochs << "\n"
            << "Training steps: "
            << history.trainLosses.size() << "\n"
            << "Elapsed seconds: "
            << elapsedSeconds << "\n"
            << "Seconds per epoch: "
            << elapsedSeconds /
            static_cast<double>(completedEpochs)
            << "\n\n"
            << "Initial training loss: "
            << initialLoss << "\n"
            << "Final training loss: "
            << finalLoss << "\n"
            << "Training improvement: "
            << trainingImprovement << "%\n\n"
            << "Initial validation loss: "
            << initialValidationLoss << "\n"
            << "Final validation loss: "
            << finalValidationLoss << "\n"
            << "Validation improvement: "
            << validationImprovement << "%\n\n"
            << "Initial learning rate: "
            << initialLearningRate << "\n"
            << "Final learning rate: "
            << finalLearningRate << "\n\n"
            << "Best checkpoint epoch: "
            << bestMetadata.epoch << "\n"
            << "Best checkpoint path: "
            << bestCheckpointPath << std::endl;

        printSection("Best-Checkpoint Generation");

        std::cout
            << "Prompt: \"To be\"\n"
            << "------------------------------------------------------------\n"
            << finalGeneratedText
            << "\n------------------------------------------------------------" << std::endl;

        /*
            These are deliberately broad success conditions.

            The shakedown proves:
              - every corpus transition was consumed;
              - training remained finite;
              - loss decreased;
              - validation remained finite;
              - checkpointing worked;
              - callback generation worked;
              - best-state restoration worked;
              - final text generation completed.
        */
        if (!(finalLoss < initialLoss)) {
            throw std::runtime_error(
                "Shakedown training loss did not decrease."
            );
        }

        if (
            bestMetadata.epoch <= 0 ||
            bestMetadata.epoch >
            static_cast<int>(completedEpochs)
            ) {
            throw std::runtime_error(
                "Best-checkpoint epoch is invalid."
            );
        }

        std::cout
            << "\n[PASS] Full dataset shakedown completed on "
            << deviceName(device)
            << std::endl;
    }

} // namespace

void runShakedownTests(Device device) {
    if (device == Device::CPU) {
        runFullDatasetShakedown(Device::CPU);
    }
    else if (device == Device::CUDA) {
        runFullDatasetShakedown(Device::CUDA);
    }
    else {
        /*
            AUTO intentionally runs both. This is appropriate for the final
            CPU-vs-CUDA validation, but explicit -C or -G remains useful when
            testing one backend at a time.
        */
        runFullDatasetShakedown(Device::CPU);

        if (isCudaAvailable()) {
            runFullDatasetShakedown(Device::CUDA);
        }
        else {
            std::cout
                << "[SKIP] CUDA unavailable; CUDA shakedown skipped." << std::endl;
        }
    }

    std::cout
        << "\n[PASS] Shakedown test suite completed." << std::endl;
}