#include "training/generation_callback.h"

#include "training/trainer.h"
#include "model/transformer.h"

#include <iostream>
#include <stdexcept>
#include <vector>

GenerationCallback::GenerationCallback(
    const Tokenizer& tokenizer_,
    const std::string& prompt_,
    const GenerationConfig& generationConfig_,
    int everyNEpochs_,
    bool printGeneratedText_,
    uint32_t randomSeed_
)
    : tokenizer(tokenizer_),
    prompt(prompt_),
    generationConfig(generationConfig_),
    everyNEpochs(everyNEpochs_),
    printGeneratedText(printGeneratedText_),
    random(randomSeed_) {

    if (everyNEpochs <= 0) {
        throw std::invalid_argument(
            "GenerationCallback everyNEpochs must be > 0."
        );
    }

    if (prompt.empty()) {
        throw std::invalid_argument(
            "GenerationCallback prompt cannot be empty."
        );
    }

    generationConfig.validate();

    // Validate immediately instead of waiting until the first epoch.
    tokenizer.encode(prompt);
}

void GenerationCallback::onEpochEnd(
    const EpochContext& context,
    TrainableModel& model,
    Optimizer& optimizer,
    TrainingHistory& history
) {
    (void)optimizer;
    (void)history;

    if (context.epoch % everyNEpochs != 0) {
        return;
    }

    Transformer* transformer =
        dynamic_cast<Transformer*>(&model);

    if (!transformer) {
        throw std::runtime_error(
            "GenerationCallback requires a Transformer model."
        );
    }

    std::vector<Parameter*> parameters =
        transformer->parameters();

    /*
        Transformer::generate currently performs its sampling and
        final-logit inspection through CPU-side tensor access.

        If training is occurring on CUDA, synchronize the trained
        parameters back to the host before generation.
    */
    if (context.device == Device::CUDA) {
        for (Parameter* parameter : parameters) {
            if (!parameter) {
                continue;
            }

            parameter->value.toCPU();
            parameter->grad.toCPU();
        }
    }

    std::string generatedText;

    try {
        generatedText = transformer->generate(
            prompt,
            tokenizer,
            generationConfig,
            random
        );
    }
    catch (...) {
        /*
            Preserve the training device even when generation fails.
        */
        if (context.device == Device::CUDA) {
            for (Parameter* parameter : parameters) {
                if (!parameter) {
                    continue;
                }

                parameter->value.toCUDA();
                parameter->grad.toCUDA();
            }
        }

        throw;
    }

    if (context.device == Device::CUDA) {
        for (Parameter* parameter : parameters) {
            if (!parameter) {
                continue;
            }

            parameter->value.toCUDA();
            parameter->grad.toCUDA();
        }
    }

    GenerationSnapshot snapshot;
    snapshot.epoch = context.epoch;
    snapshot.text = generatedText;

    snapshots.push_back(std::move(snapshot));

    if (printGeneratedText) {
        std::cout
            << "\n"
            << "==================================================\n"
            << "[GENERATION] epoch=" << context.epoch << "\n"
            << "==================================================\n"
            << generatedText
            << "\n"
            << "==================================================\n" << std::endl;
    }
}

const std::vector<GenerationSnapshot>&
GenerationCallback::getSnapshots() const {
    return snapshots;
}

const GenerationSnapshot&
GenerationCallback::getLatestSnapshot() const {
    if (snapshots.empty()) {
        throw std::runtime_error(
            "GenerationCallback has no generation snapshots."
        );
    }

    return snapshots.back();
}

void GenerationCallback::clearSnapshots() {
    snapshots.clear();
}