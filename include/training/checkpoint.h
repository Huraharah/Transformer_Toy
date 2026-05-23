#pragma once

#include "core/parameter.h"
#include "training/training_history.h"

#include <string>
#include <vector>

struct CheckpointMetadata {
    int epoch = 0;
    int globalStep = 0;
    std::string runName = "default_run";
};

class Checkpoint {
public:
    static void save(
        const std::string& path,
        const std::vector<Parameter*>& parameters,
        const CheckpointMetadata& metadata,
        const TrainingHistory& history
    );

    static void load(
        const std::string& path,
        std::vector<Parameter*>& parameters,
        CheckpointMetadata& metadata,
        TrainingHistory& history
    );
};