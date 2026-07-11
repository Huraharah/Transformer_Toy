#include "training/best_checkpoint_callback.h"
#include "training/trainer.h"

#include <iostream>
#include <vector>
#include <stdexcept>

BestCheckpointCallback::BestCheckpointCallback(
    const std::string& checkpointDirectory_,
    float minDelta_,
    bool preferValidationLoss_
)
    : checkpointDirectory(checkpointDirectory_),
    minDelta(minDelta_),
    preferValidationLoss(preferValidationLoss_),
    hasBest(false),
    bestMetric(std::numeric_limits<float>::infinity()),
    bestEpoch(0),
    bestCheckpointPath("") {

    if (minDelta < 0.0f) {
        throw std::invalid_argument("BestCheckpointCallback minDelta must be >= 0.");
    }
}

void BestCheckpointCallback::onEpochEnd(
    const EpochContext& context,
    TrainableModel& model,
    Optimizer& optimizer,
    TrainingHistory& history
) {
    (void)optimizer;

    float metric = context.trainLoss;

    if (preferValidationLoss && context.hasValidationLoss) {
        metric = context.validationLoss;
    }

    bool improved =
        !hasBest ||
        metric < bestMetric - minDelta;

    if (!improved) {
        return;
    }

    std::vector<Parameter*> params = model.parameters();

    CheckpointMetadata metadata;
    metadata.epoch = context.epoch;
    metadata.globalStep = context.globalStep;
    metadata.runName = context.runName;

    bestCheckpointPath =
        checkpointDirectory + "/" +
        context.runName +
        "_best.bin";

    if (context.device == Device::CUDA) {
        for (Parameter* p : params) {
            p->value.toCPU();
            p->grad.toCPU();
        }
    }

    Checkpoint::save(
        bestCheckpointPath,
        params,
        metadata,
        history
    );

    if (context.device == Device::CUDA) {
        for (Parameter* p : params) {
            p->value.toCUDA();
            p->grad.toCUDA();
        }
    }

    hasBest = true;
    bestMetric = metric;
    bestEpoch = context.epoch;

    std::cout
        << "[BEST] epoch=" << bestEpoch
        << " metric=" << bestMetric
        << " saved=" << bestCheckpointPath
        << "\n";
}

bool BestCheckpointCallback::hasBestCheckpoint() const {
    return hasBest;
}

float BestCheckpointCallback::getBestMetric() const {
    return bestMetric;
}

int BestCheckpointCallback::getBestEpoch() const {
    return bestEpoch;
}

const std::string& BestCheckpointCallback::getBestCheckpointPath() const {
    return bestCheckpointPath;
}