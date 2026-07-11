#pragma once

#include "training/epoch_callback.h"

#include <limits>

class EarlyStoppingCallback : public EpochCallback {
private:
    int patience;
    float minDelta;
    bool preferValidationLoss;

    bool hasBest;
    float bestMetric;
    int bestEpoch;
    int epochsWithoutImprovement;
    bool stopTraining;

public:
    EarlyStoppingCallback(
        int patience_,
        float minDelta_ = 0.0f,
        bool preferValidationLoss_ = true
    );

    void onEpochEnd(
        const EpochContext& context,
        TrainableModel& model,
        Optimizer& optimizer,
        TrainingHistory& history
    ) override;

    bool shouldStopTraining() const override;

    bool hasBestMetric() const;
    float getBestMetric() const;
    int getBestEpoch() const;
    int getEpochsWithoutImprovement() const;
};