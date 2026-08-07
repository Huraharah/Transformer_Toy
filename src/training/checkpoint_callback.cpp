#include "training/checkpoint_callback.h"
#include "training/trainer.h"

#include <string>
#include <iostream>

CheckpointCallback::CheckpointCallback(
    const std::string& checkpointDirectory_,
    int everyNEpochs_
)
    : checkpointDirectory(checkpointDirectory_),
    everyNEpochs(everyNEpochs_) {
}

void CheckpointCallback::onEpochEnd(
    const EpochContext& context,
    TrainableModel& model,
    Optimizer& optimizer,
    TrainingHistory& history
) {
    if (everyNEpochs <= 0) {
        return;
    }

    if (context.epoch % everyNEpochs != 0) {
        return;
    }

    std::vector<Parameter*> params = model.parameters();

    CheckpointMetadata metadata;
    metadata.epoch = context.epoch;
    metadata.globalStep = context.globalStep;
    metadata.runName = context.runName;

    std::string path =
        checkpointDirectory + "/" +
        context.runName +
        "_epoch_" +
        std::to_string(context.epoch) +
        ".bin";

    if (context.device == Device::CUDA) {
        for (Parameter* p : params) {
            p->value.toCPU();
            p->grad.toCPU();
        }
    }

    Checkpoint::save(path, params, metadata, history);

    if (context.device == Device::CUDA) {
        for (Parameter* p : params) {
            p->value.toCUDA();
            p->grad.toCUDA();
        }
    }

    std::cout << "[CALLBACK] Checkpoint saved: " << path << std::endl;
}