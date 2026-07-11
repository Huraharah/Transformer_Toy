#include "training/trainer.h"
#include "training/checkpoint.h"
#include "core/layer_utils.h"
#include "training/epoch_callback.h"

#include <iostream>
#include <stdexcept>
#include <string>

Trainer::Trainer(
    TrainableModel& model_,
    Loss& lossFunction_,
    Optimizer& optimizer_,
    const TrainingConfig& config_
)
    : config(config_),
    model(model_),
    lossFunction(lossFunction_),
    optimizer(optimizer_) {
}

TrainingHistory& Trainer::getHistory() {
    return history;
}

const TrainingHistory& Trainer::getHistory() const {
    return history;
}

void Trainer::train(
    const std::vector<TrainingBatch>& trainBatches,
    const std::vector<TrainingBatch>* validationBatches
) {
    //std::cout << "[DEBUG] Trainer::train entered." << std::endl;

    if (trainBatches.empty()) {
        throw std::invalid_argument("Trainer::train received empty training batches.");
    }

    if (config.epochs <= 0) {
        throw std::invalid_argument("Trainer config epochs must be > 0.");
    }

    std::vector<Parameter*> params = model.parameters();

    Device activeDevice = resolveDevice(config.device);

    for (Parameter* p : params) {
        if (activeDevice == Device::CUDA) {
            p->value.toCUDA();
            p->grad.toCUDA();
        }
        else {
            p->value.toCPU();
            p->grad.toCPU();
        }
    }

    for (int epoch = 1; epoch <= config.epochs; ++epoch) {
        float epochLossSum = 0.0f;

        //std::cout << "[DEBUG] Epoch " << epoch << " started..." << std::endl;

        for (size_t batchIndex = 0; batchIndex < trainBatches.size(); ++batchIndex) {
            TrainingBatch batch = trainBatches[batchIndex];

            if (activeDevice == Device::CUDA) {
                batch.inputs.toCUDA();
                batch.targets.toCUDA();
            }
            else {
                batch.inputs.toCPU();
                batch.targets.toCPU();
            }

            Tensor predictions = model.forward(batch.inputs);

            Tensor flatPredictions;
            Tensor flatTargets;

            if (predictions.rank() == 3) {
                flatPredictions = LayerUtils::flatten3DTo2D(predictions);

                flatTargets = Tensor({ batch.targets.size() });

                for (size_t i = 0; i < batch.targets.size(); ++i) {
                    flatTargets[i] = batch.targets[i];
                }
            }
            else if (predictions.rank() == 2) {
                flatPredictions = predictions;
                flatTargets = batch.targets;
            }
            else {
                throw std::invalid_argument(
                    "Trainer::train expects model predictions with rank 2 or 3."
                );
            }

            if (activeDevice == Device::CUDA) {
                flatTargets.toCUDA();
            }

            float lossValue = lossFunction.forward(flatPredictions, flatTargets);

            Tensor flatGradLoss = lossFunction.backward();

            Tensor gradLoss;

            if (predictions.rank() == 3) {
                gradLoss = LayerUtils::unflatten2DTo3D(
                    flatGradLoss,
                    predictions.shape()[0],
                    predictions.shape()[1]
                );
            }
            else {
                gradLoss = flatGradLoss;
            }

            model.backward(gradLoss);

            optimizer.step(params);
            optimizer.zeroGrad(params);

            history.addTrainLoss(lossValue);
            epochLossSum += lossValue;
            ++globalStep;

            if (config.logEverySteps > 0 && globalStep % config.logEverySteps == 0) {
                std::cout
                    << "[TRAIN] epoch=" << epoch
                    << " step=" << globalStep
                    << " loss=" << lossValue
                    << "\n";
            }
        }

        float avgEpochLoss = epochLossSum / static_cast<float>(trainBatches.size());

        std::cout
            << "[EPOCH] " << epoch
            << "/" << config.epochs
            << " avg_train_loss=" << avgEpochLoss;

        float valLoss = 0.0f;
        bool hasValidationLoss = false;

        if (validationBatches && !validationBatches->empty()) {
            valLoss = evaluate(*validationBatches);
            history.addValidationLoss(valLoss);
            hasValidationLoss = true;
            std::cout << " val_loss=" << valLoss;
        }

        EpochContext context;
        context.epoch = epoch;
        context.totalEpochs = config.epochs;
        context.globalStep = globalStep;
        context.trainLoss = avgEpochLoss;
        context.validationLoss = valLoss;
        context.hasValidationLoss = hasValidationLoss;
        context.device = activeDevice;
        context.runName = config.runName;
        context.checkpointDirectory = config.checkpointDirectory;
        context.checkpointEveryEpochs = config.checkpointEveryEpochs;

        std::cout << "\n";

        bool stopTraining = false;

        for (EpochCallback* callback : callbacks) {
            if (!callback) {
                continue;
            }

            callback->onEpochEnd(context, model, optimizer, history);

            if (callback->shouldStopTraining()) {
                stopTraining = true;
            }
        }

        if (stopTraining) {
            break;
        }
    }
}

float Trainer::evaluate(
    const std::vector<TrainingBatch>& validationBatches
) {
    if (validationBatches.empty()) {
        throw std::invalid_argument("Trainer::evaluate received empty validation batches.");
    }

    float totalLoss = 0.0f;

    for (TrainingBatch batch : validationBatches) {
        if (activeDevice == Device::CUDA) {
            batch.inputs.toCUDA();
            batch.targets.toCUDA();
        }
        else {
            batch.inputs.toCPU();
            batch.targets.toCPU();
        }

        Tensor predictions = model.forward(batch.inputs);

        Tensor flatPredictions;
        Tensor flatTargets;

        if (predictions.rank() == 3) {
            flatPredictions = LayerUtils::flatten3DTo2D(predictions);

            flatTargets = Tensor({ batch.targets.size() });

            for (size_t i = 0; i < batch.targets.size(); ++i) {
                flatTargets[i] = batch.targets[i];
            }

            if (activeDevice == Device::CUDA) {
                flatTargets.toCUDA();
            }
        }
        else if (predictions.rank() == 2) {
            flatPredictions = predictions;
            flatTargets = batch.targets;
        }
        else {
            throw std::invalid_argument(
                "Trainer::evaluate expects model predictions with rank 2 or 3."
            );
        }

        float lossValue = lossFunction.forward(flatPredictions, flatTargets);
        totalLoss += lossValue;
    }

    return totalLoss / static_cast<float>(validationBatches.size());
}

void Trainer::addCallback(EpochCallback* callback) {
    if (callback) {
        callbacks.push_back(callback);
    }
}

void Trainer::clearCallbacks() {
    callbacks.clear();
}