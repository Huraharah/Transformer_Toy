#pragma once

#include "training/epoch_callback.h"
#include "data/tokenizer.h"
#include "layers/config.h"
#include "core/random.h"

#include <string>
#include <vector>

struct GenerationSnapshot {
    int epoch = 0;
    std::string text;
};

class GenerationCallback : public EpochCallback {
private:
    const Tokenizer& tokenizer;

    std::string prompt;
    GenerationConfig generationConfig;

    int everyNEpochs;
    bool printGeneratedText;

    Random random;

    std::vector<GenerationSnapshot> snapshots;

public:
    GenerationCallback(
        const Tokenizer& tokenizer_,
        const std::string& prompt_,
        const GenerationConfig& generationConfig_,
        int everyNEpochs_ = 1,
        bool printGeneratedText_ = true,
        uint32_t randomSeed_ = 1234
    );

    void onEpochEnd(
        const EpochContext& context,
        TrainableModel& model,
        Optimizer& optimizer,
        TrainingHistory& history
    ) override;

    const std::vector<GenerationSnapshot>& getSnapshots() const;

    const GenerationSnapshot& getLatestSnapshot() const;

    void clearSnapshots();
};