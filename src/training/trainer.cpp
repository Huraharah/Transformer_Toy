#include "training/trainer.h"
#include "training/checkpoint.h"
#include "core/layer_utils.h"
#include "training/epoch_callback.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

Trainer::Trainer(
    TrainableModel& model_,
    Loss& lossFunction_,
    Optimizer& optimizer_,
    const TrainingConfig& config_
)
    : config(config_),
    model(model_),
    lossFunction(lossFunction_),
    optimizer(optimizer_),
    profiler_(config.enableProfiling),
    activeDevice(resolveDevice(config.device)){ 
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
    if (trainBatches.empty()) {
        throw std::invalid_argument(
            "Trainer::train received empty training batches."
        );
    }

    if (config.epochs <= 0) {
        throw std::invalid_argument(
            "Trainer config epochs must be > 0."
        );
    }

    model.train();

    /*
        Configure and reset the profiler before attaching it to the
        model hierarchy.
    */
    profiler_.setEnabled(
        config.enableProfiling
    );

    profiler_.setSynchronizeCudaPhases(
        config.enableProfiling &&
        config.synchronizeProfilingPhases
    );

    profiler_.reset();

    /*
        Attach the profiler to the model for the duration of training.

        The local guard ensures the model is detached even if training
        throws an exception.
    */
    class ModelProfilerGuard {
    private:
        TrainableModel& model_;

    public:
        ModelProfilerGuard(
            TrainableModel& model,
            TrainingProfiler* profiler
        )
            : model_(model) {
            model_.setProfiler(profiler);
        }

        ~ModelProfilerGuard() {
            model_.setProfiler(nullptr);
        }

        ModelProfilerGuard(
            const ModelProfilerGuard&
        ) = delete;

        ModelProfilerGuard& operator=(
            const ModelProfilerGuard&
            ) = delete;
    };

    ModelProfilerGuard modelProfilerGuard(
        model,
        config.enableProfiling
        ? &profiler_
        : nullptr
    );

    std::vector<Parameter*> params =
        model.parameters();

    /*
        Move model parameters to the selected device once before
        beginning the timed training loop.
    */
    for (Parameter* parameter : params) {
        if (!parameter) {
            continue;
        }

        if (activeDevice == Device::CUDA) {
            parameter->value.toCUDA();
            parameter->grad.toCUDA();
        }
        else {
            parameter->value.toCPU();
            parameter->grad.toCPU();
        }
    }

    /*
        Ensure parameter movement is complete before TotalTraining
        begins. Parameter setup is intentionally excluded from the
        training benchmark.
    */
    if (
        config.enableProfiling &&
        config.synchronizeProfilingPhases &&
        activeDevice == Device::CUDA
        ) {
        cudaSync();
    }

    profiler_.start(
        ProfilePhase::TotalTraining
    );

    for (
        int epoch = 1;
        epoch <= config.epochs;
        ++epoch
        ) {
        profiler_.start(
            ProfilePhase::Epoch
        );

        float epochLossSum = 0.0f;

        for (
            std::size_t batchIndex = 0;
            batchIndex < trainBatches.size();
            ++batchIndex
            ) {
            profiler_.start(
                ProfilePhase::TrainingStep
            );

            /*
                This remains a copy because moving tensors between
                devices mutates their device state.

                It should be revisited during the allocation/copy
                optimization pass.
            */
            TrainingBatch batch =
                trainBatches[batchIndex];

            // ====================================================
            // Batch transfer
            // ====================================================

            profiler_.start(
                ProfilePhase::BatchTransfer
            );

            if (activeDevice == Device::CUDA) {
                batch.inputs.toCUDA();
                batch.targets.toCUDA();
            }
            else {
                batch.inputs.toCPU();
                batch.targets.toCPU();
            }

            synchronizeProfilePhase(
                activeDevice
            );

            profiler_.stop(
                ProfilePhase::BatchTransfer
            );

            // ====================================================
            // Model forward
            // ====================================================

            profiler_.start(
                ProfilePhase::ModelForward
            );

            Tensor predictions =
                model.forward(batch.inputs);

            synchronizeProfilePhase(
                activeDevice
            );

            profiler_.stop(
                ProfilePhase::ModelForward
            );

            // ====================================================
            // Prepare flattened loss inputs
            // ====================================================

            Tensor flatPredictions;
            Tensor flatTargets;

            if (predictions.rank() == 3) {
                flatPredictions =
                    LayerUtils::flatten3DTo2D(
                        predictions
                    );

                flatTargets =
                    Tensor({
                        batch.targets.size()
                        });

                for (
                    std::size_t index = 0;
                    index < batch.targets.size();
                    ++index
                    ) {
                    flatTargets[index] =
                        batch.targets[index];
                }
            }
            else if (predictions.rank() == 2) {
                flatPredictions =
                    predictions;

                flatTargets =
                    batch.targets;
            }
            else {
                throw std::invalid_argument(
                    "Trainer::train expects model predictions "
                    "with rank 2 or 3."
                );
            }

            if (activeDevice == Device::CUDA) {
                flatTargets.toCUDA();
            }
            else {
                flatTargets.toCPU();
            }

            // ====================================================
            // Loss forward
            // ====================================================

            profiler_.start(
                ProfilePhase::LossForward
            );

            const float lossValue =
                lossFunction.forward(
                    flatPredictions,
                    flatTargets
                );

            synchronizeProfilePhase(
                activeDevice
            );

            profiler_.stop(
                ProfilePhase::LossForward
            );

            // ====================================================
            // Loss backward
            // ====================================================

            profiler_.start(
                ProfilePhase::LossBackward
            );

            Tensor flatGradLoss =
                lossFunction.backward();

            synchronizeProfilePhase(
                activeDevice
            );

            profiler_.stop(
                ProfilePhase::LossBackward
            );

            Tensor gradLoss;

            if (predictions.rank() == 3) {
                gradLoss =
                    LayerUtils::unflatten2DTo3D(
                        flatGradLoss,
                        predictions.shape()[0],
                        predictions.shape()[1]
                    );
            }
            else {
                gradLoss =
                    flatGradLoss;
            }

            // ====================================================
            // Model backward
            // ====================================================

            profiler_.start(
                ProfilePhase::ModelBackward
            );

            model.backward(
                gradLoss
            );

            synchronizeProfilePhase(
                activeDevice
            );

            profiler_.stop(
                ProfilePhase::ModelBackward
            );

            // ====================================================
            // Optimizer
            // ====================================================

            profiler_.start(
                ProfilePhase::OptimizerStep
            );

            optimizer.step(
                params
            );

            synchronizeProfilePhase(
                activeDevice
            );

            profiler_.stop(
                ProfilePhase::OptimizerStep
            );

            // ====================================================
            // Zero gradients
            // ====================================================

            profiler_.start(
                ProfilePhase::ZeroGrad
            );

            optimizer.zeroGrad(
                params
            );

            synchronizeProfilePhase(
                activeDevice
            );

            profiler_.stop(
                ProfilePhase::ZeroGrad
            );

            // ====================================================
            // Bookkeeping
            // ====================================================

            history.addTrainLoss(
                lossValue
            );

            epochLossSum += lossValue;
            ++globalStep;

            profiler_.incrementProcessedSteps();

            profiler_.addProcessedTokens(
                batch.inputs.size()
            );

            /*
                No synchronization is needed here. Every CUDA-producing
                child phase has already been synchronized.
            */
            profiler_.stop(
                ProfilePhase::TrainingStep
            );

            if (
                config.logEverySteps > 0 &&
                globalStep %
                config.logEverySteps == 0
                ) {
                std::cout
                    << "[TRAIN] epoch="
                    << epoch
                    << " step="
                    << globalStep
                    << " loss="
                    << lossValue
                    << std::endl;
            }
        }

        const float avgEpochLoss =
            epochLossSum /
            static_cast<float>(
                trainBatches.size()
                );

        // ========================================================
        // Validation
        // ========================================================

        float valLoss = 0.0f;
        bool hasValidationLoss = false;

        if (
            validationBatches &&
            !validationBatches->empty()
            ) {
            profiler_.start(
                ProfilePhase::Validation
            );

            valLoss =
                evaluate(
                    *validationBatches
                );

            synchronizeProfilePhase(
                activeDevice
            );

            profiler_.stop(
                ProfilePhase::Validation
            );

            history.addValidationLoss(
                valLoss
            );

            hasValidationLoss = true;
        }

        std::cout
            << "[EPOCH] "
            << epoch
            << "/"
            << config.epochs
            << " avg_train_loss="
            << avgEpochLoss;

        if (hasValidationLoss) {
            std::cout
                << " val_loss="
                << valLoss;
        }

        std::cout << std::endl;

        EpochContext context;
        context.epoch = epoch;
        context.totalEpochs =
            config.epochs;
        context.globalStep =
            globalStep;
        context.trainLoss =
            avgEpochLoss;
        context.validationLoss =
            valLoss;
        context.hasValidationLoss =
            hasValidationLoss;
        context.device =
            activeDevice;
        context.runName =
            config.runName;
        context.checkpointDirectory =
            config.checkpointDirectory;
        context.checkpointEveryEpochs =
            config.checkpointEveryEpochs;

        // ========================================================
        // Callbacks
        // ========================================================

        bool stopTraining = false;

        profiler_.start(
            ProfilePhase::Callbacks
        );

        for (EpochCallback* callback : callbacks) {
            if (!callback) {
                continue;
            }

            callback->onEpochEnd(
                context,
                model,
                optimizer,
                history
            );

            if (
                callback->shouldStopTraining()
                ) {
                stopTraining = true;
            }
        }

        synchronizeProfilePhase(
            activeDevice
        );

        profiler_.stop(
            ProfilePhase::Callbacks
        );

        /*
            The epoch contains only already-synchronized child phases,
            so another synchronization is unnecessary.
        */
        profiler_.stop(
            ProfilePhase::Epoch
        );

        if (stopTraining) {
            break;
        }
    }

    /*
        All CUDA-producing phases have already synchronized individually.
    */
    profiler_.stop(
        ProfilePhase::TotalTraining
    );

    if (
        config.enableProfiling &&
        config.printProfileSummary
        ) {
        profiler_.printSummary(
            std::cout
        );
    }
}

float Trainer::evaluate(
    const std::vector<TrainingBatch>& validationBatches
) {
    if (validationBatches.empty()) {
        throw std::invalid_argument("Trainer::evaluate received empty validation batches.");
    }

    const bool wasTraining = model.isTraining();

    model.eval();

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

    if (wasTraining) {
        model.train();
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

const TrainingProfiler&
Trainer::getProfiler() const {
    return profiler_;
}


TrainingProfiler&
Trainer::getProfiler() {
    return profiler_;
}

void Trainer::synchronizeProfilePhase(Device activeDevice) const {
    if (!config.enableProfiling || !config.synchronizeProfilingPhases || activeDevice != Device::CUDA) {
        return;
    }

    const cudaError_t status =
        cudaDeviceSynchronize();

    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(
                "CUDA profiling synchronization failed: "
            ) +
            cudaGetErrorString(status)
        );
    }
}