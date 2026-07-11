#pragma once

#include "training/epoch_callback.h"
#include "training/checkpoint.h"

#include <string>
#include <limits>

class BestCheckpointCallback : public EpochCallback {
private:
    std::string checkpointDirectory;
    float minDelta;
    bool preferValidationLoss;

    bool hasBest;
    float bestMetric;
    int bestEpoch;
    std::string bestCheckpointPath;

public:
    BestCheckpointCallback(
        const std::string& checkpointDirectory_,
        float minDelta_ = 0.0f,
        bool preferValidationLoss_ = true
    );

    void onEpochEnd(
        const EpochContext& context,
        TrainableModel& model,
        Optimizer& optimizer,
        TrainingHistory& history
    ) override;

    bool hasBestCheckpoint() const;
    float getBestMetric() const;
    int getBestEpoch() const;
    const std::string& getBestCheckpointPath() const;
};
