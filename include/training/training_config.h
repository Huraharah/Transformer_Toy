#pragma once

#include <string>

struct TrainingConfig {
    int epochs = 1;
    int batchSize = 1;

    float learningRate = 0.001f;

    int logEverySteps = 10;
    int checkpointEveryEpochs = 1;

    bool useValidation = false;
    bool shuffle = true;

    std::string checkpointDirectory = "checkpoints";
    std::string runName = "default_run";
};