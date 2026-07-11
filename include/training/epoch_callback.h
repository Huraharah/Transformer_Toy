#pragma once

#include "core/tensor.h"

#include <string>

class TrainableModel;
class Optimizer;
struct TrainingHistory;

struct EpochContext {
    int epoch = 0;
    int totalEpochs = 0;
    int globalStep = 0;

    float trainLoss = 0.0f;
    float validationLoss = 0.0f;
    bool hasValidationLoss = false;

    Device device = Device::CPU;
    std::string runName;
    int checkpointEveryEpochs = 1;
    std::string checkpointDirectory;
};

class EpochCallback {
public:
    virtual ~EpochCallback() = default;

    virtual void onEpochEnd(
        const EpochContext& context,
        TrainableModel& model,
        Optimizer& optimizer,
        TrainingHistory& history
    ) = 0;

    virtual bool shouldStopTraining() const {
        return false;
    }
};