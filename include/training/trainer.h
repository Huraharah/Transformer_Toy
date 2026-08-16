#pragma once

#include "core/tensor.h"
#include "core/parameter.h"
#include "core/cuda_utils.h"
#include "training/loss.h"
#include "training/optimizer.h"
#include "training/training_config.h"
#include "training/training_history.h"
#include "training/epoch_callback.h"
#include "training/training_profiler.h"

#include <vector>

class TrainableModel {
public:
    virtual ~TrainableModel() = default;

    virtual void setProfiler(TrainingProfiler* profiler) {};

    virtual Tensor forward(const Tensor& inputs) = 0;
    virtual void backward(const Tensor& gradOutput) = 0;
    virtual std::vector<Parameter*> parameters() = 0;
    
    virtual void train() = 0;
    virtual void eval() = 0;
    virtual bool isTraining() const = 0;
};

struct TrainingBatch {
    Tensor inputs;
    Tensor targets;
};

class Trainer {
private:
    TrainingConfig config;
    TrainableModel& model;
    Loss& lossFunction;
    Optimizer& optimizer;
    TrainingHistory history;
    TrainingProfiler profiler_;

    Device activeDevice = resolveDevice(config.device);
    void synchronizeProfilePhase(Device activeDevice) const;

    std::vector<EpochCallback*> callbacks;

    int globalStep = 0;

public:
    Trainer(
        TrainableModel& model_,
        Loss& lossFunction_,
        Optimizer& optimizer_,
        const TrainingConfig& config_
    );

    TrainingHistory& getHistory();
    const TrainingHistory& getHistory() const;

    void train(
        const std::vector<TrainingBatch>& trainBatches,
        const std::vector<TrainingBatch>* validationBatches = nullptr
    );

    float evaluate(const std::vector<TrainingBatch>& validationBatches);

    void addCallback(EpochCallback* callback);
    void clearCallbacks();

    const TrainingProfiler& getProfiler() const;
    TrainingProfiler& getProfiler();
};