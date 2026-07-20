#pragma once
#include "core/tensor.h"
#include "layers/config.h"
#include "training/learning_rate_scheduler_callback.h"

#include <string>

struct TrainingConfig {
    // Core training
    int epochs = 1;
    int batchSize = 1;
    Device device = Device::AUTO;

    // Logging
    int logEverySteps = 0;
    std::string runName = "default_run";

    // Checkpointing
    bool enableCheckpointing = false;
    bool enableBestCheckpoint = false;
    int checkpointEveryEpochs = 1;
    float bestCheckpointMinDelta = 0.0f;
    std::string checkpointDirectory = "checkpoints";

    // Early stopping
    bool enableEarlyStopping = false;
    int earlyStoppingPatience = 5;
    float earlyStoppingMinDelta = 0.0f;
    bool preferValidationLoss = true;

    // Learning-rate scheduler
    bool enableLRScheduler = false;
    LearningRateSchedule lrSchedule =
        LearningRateSchedule::StepDecay; // match your actual intended default
    int lrStepSize = 1;
    float lrGamma = 1.0f;
    float minimumLearningRate = 0.0f;

    // Generation snapshots
    bool enableGenerationSnapshots = false;
    int generationEveryEpochs = 1;
    GenerationConfig generationConfig{};
    std::string generationPrompt;

    // Profiling
    bool enableProfiling = false;
    bool printProfileSummary = true;
    bool synchronizeProfilingPhases = false;
    int profileWarmupSteps = 5;
    int profileEverySteps = 1;

    void validate() const {
        if (epochs <= 0) {
            throw std::invalid_argument(
                "TrainingConfig: epochs must be greater than zero."
            );
        }

        if (batchSize <= 0) {
            throw std::invalid_argument(
                "TrainingConfig: batchSize must be greater than zero."
            );
        }

        if (
            enableCheckpointing &&
            checkpointEveryEpochs <= 0
            ) {
            throw std::invalid_argument(
                "TrainingConfig: checkpointEveryEpochs must be "
                "greater than zero when checkpointing is enabled."
            );
        }

        if (
            enableEarlyStopping &&
            earlyStoppingPatience <= 0
            ) {
            throw std::invalid_argument(
                "TrainingConfig: earlyStoppingPatience must be "
                "greater than zero when early stopping is enabled."
            );
        }

        if (
            enableGenerationSnapshots &&
            generationEveryEpochs <= 0
            ) {
            throw std::invalid_argument(
                "TrainingConfig: generationEveryEpochs must be "
                "greater than zero when generation snapshots are enabled."
            );
        }

        generationConfig.validate();
    }
};