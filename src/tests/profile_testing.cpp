#include "tests/profile_testing.h"
#include "core/cuda_utils.h"
#include "core/random.h"
#include "layers/config.h"
#include "data/dataset.h"
#include "data/tokenizer.h"
#include "training/cross_entropy_loss.h"
#include "model/transformer.h"
#include "training/adam_optimizer.h"
#include "training/trainer.h"
#include "training/training_config.h"

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

    /*
        Restores the prior CUDA synchronization state even if the
        benchmark throws an exception.
    */
    class ScopedCudaSynchronization {
    private:
        bool previousState_;

    public:
        explicit ScopedCudaSynchronization(
            bool enabled
        )
            : previousState_(
                isCudaSynchronizationEnabled()
            ) {
            setCudaSynchronizationEnabled(
                enabled
            );
        }

        ~ScopedCudaSynchronization() {
            setCudaSynchronizationEnabled(
                previousState_
            );
        }

        ScopedCudaSynchronization(
            const ScopedCudaSynchronization&
        ) = delete;

        ScopedCudaSynchronization& operator=(
            const ScopedCudaSynchronization&
            ) = delete;
    };


    std::vector<TrainingBatch> buildBenchmarkBatches(
        const TextDataset& dataset,
        std::size_t numberOfBatches,
        std::size_t batchSize,
        Random& rng
    ) {
        std::vector<TrainingBatch> batches;
        batches.reserve(numberOfBatches);

        for (
            std::size_t batchIndex = 0;
            batchIndex < numberOfBatches;
            ++batchIndex
            ) {
            Tensor inputs;
            Tensor targets;

            dataset.getBatch(
                batchSize,
                rng,
                inputs,
                targets
            );

            batches.push_back({
                std::move(inputs),
                std::move(targets)
                });
        }

        return batches;
    }


    const char* deviceName(
        Device device
    ) {
        switch (device) {
        case Device::CPU:
            return "CPU";

        case Device::CUDA:
            return "CUDA";

        case Device::AUTO:
            return "AUTO";

        default:
            return "Unknown";
        }
    }

} // namespace


void runProfileBenchmark(
    Device requestedDevice
) {
    constexpr std::size_t contextLength = 64;
    constexpr std::size_t batchSize = 16;

    constexpr std::size_t warmupSteps = 10;
    constexpr std::size_t measuredSteps = 100;

    constexpr int dModel = 64;
    constexpr int dFF = 128;
    constexpr int numLayers = 2;
    constexpr int numHeads = 4;

    const Device activeDevice =
        resolveDevice(requestedDevice);

    /*
        This first benchmark intentionally enables synchronization
        inside every CUDA launch wrapper.

        It measures the legacy behavior that the project used during
        the shakedown and tokenizer smoke tests.
    */
    ScopedCudaSynchronization synchronizationScope(
        activeDevice == Device::CUDA ? false : isCudaSynchronizationEnabled()
    );

    std::cout
        << "\n"
        << "============================================================\n"
        << "||           Phase-Synchronized CUDA Baseline             ||\n"
        << "============================================================\n";

    if (
        requestedDevice == Device::CUDA &&
        activeDevice != Device::CUDA
        ) {
        throw std::runtime_error(
            "CUDA profiling benchmark requested, but CUDA "
            "was not available."
        );
    }

    std::filesystem::create_directories(
        "./test_tokenizers"
    );

    TokenizerConfig tokenizerConfig;
    tokenizerConfig.type =
        TokenizerType::BPE;
    tokenizerConfig.vocabSize = 128;
    tokenizerConfig.minFrequency = 2;
    tokenizerConfig.modelPath =
        "./test_tokenizers/profile_bpe_128.tok";
    tokenizerConfig.trainIfMissing = true;

    /*
        Tokenizer loading/training and corpus encoding happen before
        timed training begins, so they are not part of the profile.
    */
    TextDataset dataset(
        "data/shakespeare.txt",
        contextLength,
        tokenizerConfig
    );

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
        static_cast<int>(
            dataset.vocabSize()
            );
    modelConfig.max_seq_len =
        static_cast<int>(
            dataset.contextLength()
            );
    modelConfig.num_layers = numLayers;
    modelConfig.block = blockConfig;
    modelConfig.learned_positional_embeddings =
        true;
    modelConfig.embedding_dropout = 0.0f;

    modelConfig.validate();

    Random rng(42);

    Transformer model(
        modelConfig,
        rng
    );

    CrossEntropyLoss loss;
    AdamOptimizer optimizer(0.001f);

    std::vector<TrainingBatch> warmupBatches =
        buildBenchmarkBatches(
            dataset,
            warmupSteps,
            batchSize,
            rng
        );

    std::vector<TrainingBatch> measuredBatches =
        buildBenchmarkBatches(
            dataset,
            measuredSteps,
            batchSize,
            rng
        );

    std::cout
        << "[BENCHMARK CONFIGURATION]\n"
        << "  mode: Trainer phase-boundary synchronization\n"
        << "  device: "
        << deviceName(activeDevice)
        << "\n"
        << "  build expectation: Release\n"
        << "  tokenizer: BPE\n"
        << "  vocabulary: "
        << dataset.vocabSize()
        << "\n"
        << "  raw characters: "
        << dataset.rawText().size()
        << "\n"
        << "  encoded tokens: "
        << dataset.tokenCount()
        << "\n"
        << "  compression ratio: "
        << static_cast<double>(
            dataset.rawText().size()
            ) /
        static_cast<double>(
            dataset.tokenCount()
            )
        << "\n"
        << "  context length: "
        << contextLength
        << "\n"
        << "  batch size: "
        << batchSize
        << "\n"
        << "  tokens per step: "
        << batchSize * contextLength
        << "\n"
        << "  d_model: "
        << dModel
        << "\n"
        << "  d_ff: "
        << dFF
        << "\n"
        << "  layers: "
        << numLayers
        << "\n"
        << "  heads: "
        << numHeads
        << "\n"
        << "  warmup steps: "
        << warmupSteps
        << "\n"
        << "  measured steps: "
        << measuredSteps
        << "\n"
        << "  wrapper synchronization: "
        << (
            isCudaSynchronizationEnabled()
            ? "enabled"
            : "disabled"
            )
        << "\n";

    // ========================================================
    // Warmup
    // ========================================================

    TrainingConfig warmupConfig;
    warmupConfig.epochs = 1;
    warmupConfig.device = activeDevice;
    warmupConfig.logEverySteps = 0;

    warmupConfig.enableProfiling = false;
    warmupConfig.printProfileSummary = false;

    /*
        Trainer-level phase synchronization is deliberately disabled.
        Synchronization currently occurs inside every CUDA wrapper.
    */
    warmupConfig.synchronizeProfilingPhases =
        false;

    std::cout
        << "\n[BENCHMARK] Beginning warmup..."
        << std::endl;

    {
        Trainer warmupTrainer(
            model,
            loss,
            optimizer,
            warmupConfig
        );

        warmupTrainer.train(
            warmupBatches,
            nullptr
        );
    }

    if (activeDevice == Device::CUDA) {
        cudaSync();
    }

    std::cout
        << "[BENCHMARK] Warmup complete."
        << std::endl;

    // ========================================================
    // Measured phase-synchronized pass
    // ========================================================

    TrainingConfig benchmarkConfig;
    benchmarkConfig.epochs = 1;
    benchmarkConfig.device = activeDevice;
    benchmarkConfig.logEverySteps = 0;

    benchmarkConfig.enableProfiling = true;
    benchmarkConfig.printProfileSummary = true;

    /*
        Keep this false for the legacy baseline.

        The CUDA wrappers already call cudaSyncIfEnabled(), and the
        ScopedCudaSynchronization object enabled that mechanism.
        Enabling Trainer synchronization as well would double-sync.
    */
    benchmarkConfig.synchronizeProfilingPhases =
        true;

    std::cout
        << "\n[BENCHMARK] Beginning measured phased-synchronized pass..."
        << std::endl;

    Trainer benchmarkTrainer(
        model,
        loss,
        optimizer,
        benchmarkConfig
    );

    benchmarkTrainer.train(
        measuredBatches,
        nullptr
    );

    const TrainingProfiler& profiler =
        benchmarkTrainer.getProfiler();

    if (
        profiler.stats(
            ProfilePhase::TrainingStep
        ).callCount != measuredSteps
        ) {
        throw std::runtime_error(
            "Profile benchmark did not record the expected "
            "number of training steps."
        );
    }

    if (
        profiler.totalTrainingMilliseconds() <= 0.0
        ) {
        throw std::runtime_error(
            "Profile benchmark recorded no training duration."
        );
    }

    if (
        profiler.stepsPerSecond() <= 0.0 ||
        profiler.tokensPerSecond() <= 0.0
        ) {
        throw std::runtime_error(
            "Profile benchmark produced invalid throughput values."
        );
    }

    std::cout
        << "\n"
        << "[BENCHMARK RESULT]\n"
        << "  recorded steps: "
        << profiler.stats(
            ProfilePhase::TrainingStep
        ).callCount
        << "\n"
        << "  total training ms: "
        << profiler.totalTrainingMilliseconds()
        << "\n"
        << "  steps per second: "
        << profiler.stepsPerSecond()
        << "\n"
        << "  tokens per second: "
        << profiler.tokensPerSecond()
        << "\n"
        << "\n"
        << "[PASS] Synchronized profiling baseline completed.\n";
}