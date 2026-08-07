#include "training/early_stopping_callback.h"

#include <iostream>
#include <stdexcept>

EarlyStoppingCallback::EarlyStoppingCallback(
    int patience_,
    float minDelta_,
    bool preferValidationLoss_
)
    : patience(patience_),
    minDelta(minDelta_),
    preferValidationLoss(preferValidationLoss_),
    hasBest(false),
    bestMetric(std::numeric_limits<float>::infinity()),
    bestEpoch(0),
    epochsWithoutImprovement(0),
    stopTraining(false) {

    if (patience < 0) {
        throw std::invalid_argument("EarlyStoppingCallback patience must be >= 0.");
    }

    if (minDelta < 0.0f) {
        throw std::invalid_argument("EarlyStoppingCallback minDelta must be >= 0.");
    }
}

void EarlyStoppingCallback::onEpochEnd(
    const EpochContext& context,
    TrainableModel& model,
    Optimizer& optimizer,
    TrainingHistory& history
) {
    (void)model;
    (void)optimizer;
    (void)history;

    float metric = context.trainLoss;

    if (preferValidationLoss && context.hasValidationLoss) {
        metric = context.validationLoss;
    }

    bool improved =
        !hasBest ||
        metric < bestMetric - minDelta;

    if (improved) {
        hasBest = true;
        bestMetric = metric;
        bestEpoch = context.epoch;
        epochsWithoutImprovement = 0;

        std::cout
            << "[EARLY_STOP] improvement epoch=" << bestEpoch
            << " metric=" << bestMetric
            << std::endl;

        return;
    }

    ++epochsWithoutImprovement;

    std::cout
        << "[EARLY_STOP] no_improvement epoch=" << context.epoch
        << " bad_epochs=" << epochsWithoutImprovement
        << "/" << patience
        << " best_epoch=" << bestEpoch
        << " best_metric=" << bestMetric
        << std::endl;

    if (epochsWithoutImprovement >= patience) {
        stopTraining = true;

        std::cout
            << "[EARLY_STOP] stopping at epoch=" << context.epoch
            << " best_epoch=" << bestEpoch
            << " best_metric=" << bestMetric
            << std::endl;
    }
}

bool EarlyStoppingCallback::shouldStopTraining() const {
    return stopTraining;
}

bool EarlyStoppingCallback::hasBestMetric() const {
    return hasBest;
}

float EarlyStoppingCallback::getBestMetric() const {
    return bestMetric;
}

int EarlyStoppingCallback::getBestEpoch() const {
    return bestEpoch;
}

int EarlyStoppingCallback::getEpochsWithoutImprovement() const {
    return epochsWithoutImprovement;
}