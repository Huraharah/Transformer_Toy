#include "training/learning_rate_scheduler_callback.h"
#include "training/optimizer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
    constexpr float PI = 3.14159265358979323846f;
}

LearningRateSchedulerCallback::LearningRateSchedulerCallback(
    LearningRateSchedule schedule_,
    int stepSize_,
    float gamma_,
    float minimumLearningRate_
)
    : schedule(schedule_),
    stepSize(stepSize_),
    gamma(gamma_),
    minimumLearningRate(minimumLearningRate_),
    initialized(false),
    initialLearningRate(0.0f),
    currentLearningRate(0.0f) {

    if (stepSize <= 0) {
        throw std::invalid_argument(
            "LearningRateSchedulerCallback stepSize must be > 0."
        );
    }

    if (gamma <= 0.0f) {
        throw std::invalid_argument(
            "LearningRateSchedulerCallback gamma must be > 0."
        );
    }

    if (minimumLearningRate < 0.0f) {
        throw std::invalid_argument(
            "LearningRateSchedulerCallback minimumLearningRate must be >= 0."
        );
    }
}

float LearningRateSchedulerCallback::calculateLearningRate(
    const EpochContext& context
) const {
    float newLearningRate = initialLearningRate;

    switch (schedule) {
    case LearningRateSchedule::Constant:
        newLearningRate = initialLearningRate;
        break;

    case LearningRateSchedule::StepDecay: {
        int decayCount = context.epoch / stepSize;

        newLearningRate =
            initialLearningRate *
            std::pow(gamma, static_cast<float>(decayCount));

        break;
    }

    case LearningRateSchedule::ExponentialDecay:
        newLearningRate =
            initialLearningRate *
            std::pow(gamma, static_cast<float>(context.epoch));
        break;

    case LearningRateSchedule::CosineDecay: {
        if (context.totalEpochs <= 0) {
            throw std::runtime_error(
                "Cosine decay requires totalEpochs > 0."
            );
        }

        float progress =
            static_cast<float>(context.epoch) /
            static_cast<float>(context.totalEpochs);

        progress = std::clamp(progress, 0.0f, 1.0f);

        float cosineFactor =
            0.5f * (1.0f + std::cos(PI * progress));

        newLearningRate =
            minimumLearningRate +
            (initialLearningRate - minimumLearningRate) *
            cosineFactor;

        break;
    }

    default:
        throw std::runtime_error(
            "Unknown learning-rate schedule."
        );
    }

    return std::max(newLearningRate, minimumLearningRate);
}

void LearningRateSchedulerCallback::onEpochEnd(
    const EpochContext& context,
    TrainableModel& model,
    Optimizer& optimizer,
    TrainingHistory& history
) {
    (void)model;
    (void)history;

    if (!initialized) {
        initialLearningRate = optimizer.getLearningRate();
        currentLearningRate = initialLearningRate;
        initialized = true;
    }

    float previousLearningRate = optimizer.getLearningRate();

    float newLearningRate =
        calculateLearningRate(context);

    optimizer.setLearningRate(newLearningRate);
    currentLearningRate = newLearningRate;

    std::cout
        << "[LR_SCHEDULER] epoch=" << context.epoch
        << " old_lr=" << previousLearningRate
        << " new_lr=" << newLearningRate
        << "\n";
}

float LearningRateSchedulerCallback::getInitialLearningRate() const {
    return initialLearningRate;
}

float LearningRateSchedulerCallback::getCurrentLearningRate() const {
    return currentLearningRate;
}