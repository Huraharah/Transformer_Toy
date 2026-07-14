#include "training/training_session.h"

#include <memory>
#include <iostream>

TrainingSession::TrainingSession(
    TrainableModel& model,
    Loss& lossFunction,
    Optimizer& optimizer,
    const TrainingConfig& config,
    const Tokenizer* tokenizer
)
    : trainer(
        model,
        lossFunction,
        optimizer,
        config
    ) {

    if (config.enableCheckpointing) {
        checkpointCallback =
            std::make_unique<CheckpointCallback>(
                config.checkpointDirectory,
                config.checkpointEveryEpochs
            );

        trainer.addCallback(checkpointCallback.get());
    }

    if (config.enableBestCheckpoint) {
        bestCheckpointCallback =
            std::make_unique<BestCheckpointCallback>(
                config.checkpointDirectory,
                config.bestCheckpointMinDelta,
                config.preferValidationLoss
            );

        trainer.addCallback(bestCheckpointCallback.get());
    }

    if (config.enableEarlyStopping) {
        earlyStoppingCallback =
            std::make_unique<EarlyStoppingCallback>(
                config.earlyStoppingPatience,
                config.earlyStoppingMinDelta,
                config.preferValidationLoss
            );

        trainer.addCallback(earlyStoppingCallback.get());
    }

    if (config.enableLRScheduler) {
        lrSchedulerCallback =
            std::make_unique<LearningRateSchedulerCallback>(
                config.lrSchedule,
                config.lrStepSize,
                config.lrGamma,
                config.minimumLearningRate
            );

        trainer.addCallback(lrSchedulerCallback.get());
    }

    if (config.enableGenerationSnapshots) {
        if (!tokenizer) {
            throw std::invalid_argument(
                "Generation snapshots are enabled, but no tokenizer was provided."
            );
        }

        generationCallback =
            std::make_unique<GenerationCallback>(
                *tokenizer,
                config.generationPrompt,
                config.generationConfig,
                config.generationEveryEpochs,
                true,
                config.generationConfig.randomSeed
            );

        trainer.addCallback(generationCallback.get());
    }
}



void TrainingSession::train(
    const std::vector<TrainingBatch>& trainBatches,
    const std::vector<TrainingBatch>* validationBatches
) {
    //std::cout << "[DEBUG] TrainingSession::train entered...calling trainer.train()" << std::endl;

    trainer.train(
        trainBatches,
        validationBatches
    );
}

TrainingHistory& TrainingSession::getHistory() {
    return trainer.getHistory();
}

const TrainingHistory& TrainingSession::getHistory() const {
    return trainer.getHistory();
}

GenerationCallback* TrainingSession::getGenerationCallback() {
    return generationCallback.get();
}

const GenerationCallback* TrainingSession::getGenerationCallback() const {
    return generationCallback.get();
}