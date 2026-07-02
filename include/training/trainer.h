#pragma once

#include "core/tensor.h"
#include "core/parameter.h"
#include "core/cuda_utils.h"
#include "training/loss.h"
#include "training/optimizer.h"
#include "training/training_config.h"
#include "training/training_history.h"

#include <vector>

class TrainableModel {
public:
    virtual ~TrainableModel() = default;

    virtual Tensor forward(const Tensor& inputs) = 0;
    virtual void backward(const Tensor& gradOutput) = 0;
    virtual std::vector<Parameter*> parameters() = 0;
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

    Device activeDevice = resolveDevice(config.device);

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
};