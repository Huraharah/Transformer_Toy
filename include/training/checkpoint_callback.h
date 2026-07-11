#pragma once

#include "training/epoch_callback.h"
#include "training/checkpoint.h"

#include <string>

class CheckpointCallback : public EpochCallback {
private:
    std::string checkpointDirectory;
    int everyNEpochs;

public:
    CheckpointCallback(
        const std::string& checkpointDirectory_,
        int everyNEpochs_
    );

    void onEpochEnd(
        const EpochContext& context,
        TrainableModel& model,
        Optimizer& optimizer,
        TrainingHistory& history
    ) override;
};