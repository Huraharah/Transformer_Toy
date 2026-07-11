#pragma once
#include "core/tensor.h"
#include "layers/config.h"
#include "training/learning_rate_scheduler_callback.h"

#include <string>

struct TrainingConfig {

    //-------------------------
    // Core Training
    //-------------------------

    int epochs;
    int batchSize;
    Device device;

    //-------------------------
    // Logging
    //-------------------------

    int logEverySteps;
    std::string runName;

    //-------------------------
    // Checkpointing
    //-------------------------

    bool enableCheckpointing;
    bool enableBestCheckpoint;
    int checkpointEveryEpochs;
    float bestCheckpointMinDelta;
    std::string checkpointDirectory;

    //-------------------------
    // Early Stopping
    //-------------------------

    bool enableEarlyStopping;
    int earlyStoppingPatience;
    float earlyStoppingMinDelta;
    bool preferValidationLoss;

    //-------------------------
    // Learning Rate Scheduler
    //-------------------------

    bool enableLRScheduler;
    LearningRateSchedule lrSchedule;
    int lrStepSize;
    float lrGamma;
    float minimumLearningRate;

    //-------------------------
    // Generation Snapshots
    //-------------------------

    bool enableGenerationSnapshots;
    int generationEveryEpochs;
    GenerationConfig generationConfig;
    std::string generationPrompt;
};