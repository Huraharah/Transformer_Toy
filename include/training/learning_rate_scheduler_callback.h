#pragma once

#include "training/epoch_callback.h"

enum class LearningRateSchedule {
    Constant,
    StepDecay,
    ExponentialDecay,
    CosineDecay
};

class LearningRateSchedulerCallback : public EpochCallback {
private:
    LearningRateSchedule schedule;

    int stepSize;
    float gamma;
    float minimumLearningRate;

    bool initialized;
    float initialLearningRate;
    float currentLearningRate;

    float calculateLearningRate(
        const EpochContext& context
    ) const;

public:
    LearningRateSchedulerCallback(
        LearningRateSchedule schedule_,
        int stepSize_ = 1,
        float gamma_ = 0.1f,
        float minimumLearningRate_ = 0.0f
    );

    void onEpochEnd(
        const EpochContext& context,
        TrainableModel& model,
        Optimizer& optimizer,
        TrainingHistory& history
    ) override;

    float getInitialLearningRate() const;
    float getCurrentLearningRate() const;
};
