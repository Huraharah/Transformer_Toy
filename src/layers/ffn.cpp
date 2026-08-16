#include "layers/ffn.h"
#include "core/math_utils.h"
#include "core/parameter.h"
#include "kernels/activation_kernels.cuh"
#include "training/training_profiler.h"

#include <stdexcept>
#include <utility>

FFN::FFN(size_t embedDim, size_t hiddenDim, float dropoutProbability, Random& rng)
    : embedDim_(embedDim),
    hiddenDim_(hiddenDim),
    linear1_(embedDim, hiddenDim, rng),
    linear2_(hiddenDim, embedDim, rng),
    dropout_(dropoutProbability, rng){
}

void FFN::setProfiler(TrainingProfiler* profiler) {
	profiler_ = profiler;
}

Tensor FFN::forward(
    const Tensor& input
) {
    if (input.rank() != 3) {
        throw std::invalid_argument(
            "FFN::forward expects input shape "
            "[batch, sequence, embedDim]."
        );
    }

    const size_t batchSize =
        input.shape()[0];

    const size_t sequenceLength =
        input.shape()[1];

    const size_t inputEmbedDim =
        input.shape()[2];

    if (inputEmbedDim != embedDim_) {
        throw std::invalid_argument(
            "FFN embed dimension mismatch."
        );
    }

    cachedInputShape_ = {
        batchSize,
        sequenceLength,
        embedDim_
    };

    Tensor flatInput;
    Tensor hidden;
    Tensor outputFlat;
    Tensor output;

    // ========================================================
    // Flatten input
    // ========================================================

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::FFNFlattenForward,
            input.device()
        );

        flatInput = input;
        flatInput.reshape({ batchSize * sequenceLength, embedDim_ });
    }

    // ========================================================
    // First linear projection
    // ========================================================

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::FFNLinear1Forward,
            input.device()
        );

        hidden = linear1_.forward(flatInput);
    }

    /*
        Preserve the pre-activation tensor for GELU backward.
    */
    cachedHiddenPreActivation_ = hidden;

    // ========================================================
    // GELU activation
    // ========================================================

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::FFNGeluForward,
            hidden.device()
        );

        if (hidden.device() == Device::CUDA) {
            launchGeluForward(
                hidden.deviceData(),
                hidden.size()
            );
        }
        else {
            for ( size_t index = 0; index < hidden.size(); ++index ) {
                hidden[index] = MathUtils::gelu( hidden[index] );
            }
        }
    }

    // ========================================================
    // Second linear projection
    // ========================================================

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::FFNLinear2Forward,
            hidden.device()
        );

        hidden = dropout_.forward(hidden);
        outputFlat = linear2_.forward(hidden);
    }

    // ========================================================
    // Restore [batch, sequence, embedDim]
    // ========================================================

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::FFNUnflattenForward,
            outputFlat.device()
        );

        outputFlat.reshape({ batchSize, sequenceLength, embedDim_ });
        output = std::move(outputFlat);
    }

    return output;
}

Tensor FFN::backward(
    const Tensor& gradOutput
) {
    if (gradOutput.rank() != 3) {
        throw std::invalid_argument(
            "FFN::backward expects gradOutput shape "
            "[batch, sequence, embedDim]."
        );
    }

    const size_t batchSize =
        gradOutput.shape()[0];

    const size_t sequenceLength =
        gradOutput.shape()[1];

    const size_t outputEmbedDim =
        gradOutput.shape()[2];

    if (outputEmbedDim != embedDim_) {
        throw std::invalid_argument(
            "FFN backward embed dimension mismatch."
        );
    }

    if (cachedInputShape_.empty()) {
        throw std::runtime_error(
            "FFN::backward called before forward."
        );
    }

    if (
        cachedInputShape_[0] != batchSize ||
        cachedInputShape_[1] != sequenceLength
        ) {
        throw std::invalid_argument(
            "FFN backward batch or sequence size mismatch."
        );
    }

    if (
        cachedHiddenPreActivation_.empty()
        ) {
        throw std::runtime_error(
            "FFN backward missing cached pre-activation."
        );
    }

    Tensor flatGradOutput;
    Tensor gradHidden;
    Tensor gradInputFlat;
    Tensor gradInput;

    // ========================================================
    // Flatten output gradient
    // ========================================================

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::FFNFlattenBackward,
            gradOutput.device()
        );

        flatGradOutput = gradOutput;
        flatGradOutput.reshape({ batchSize * sequenceLength, embedDim_ });
    }

    // ========================================================
    // Second linear layer backward
    // ========================================================

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::FFNLinear2Backward,
            flatGradOutput.device()
        );

        gradHidden = linear2_.backward( flatGradOutput);
        gradHidden = dropout_.backward(gradHidden);
    }

    if (
        cachedHiddenPreActivation_.size() !=
        gradHidden.size()
        ) {
        throw std::runtime_error(
            "FFN backward cached activation size mismatch."
        );
    }

    // ========================================================
    // GELU backward
    // ========================================================

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::FFNGeluBackward,
            gradHidden.device()
        );

        if (gradHidden.device() == Device::CUDA) {
            launchGeluBackward(
                cachedHiddenPreActivation_
                .deviceData(),
                gradHidden.deviceData(),
                gradHidden.size()
            );
        }
        else {
            for ( size_t index = 0; index < gradHidden.size(); ++index ) {
                const float activation = cachedHiddenPreActivation_[index];

                const float geluGradient = MathUtils::geluDerivative(activation);

                gradHidden[index] *= geluGradient;
            }
        }
    }

    // ========================================================
    // First linear layer backward
    // ========================================================

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::FFNLinear1Backward,
            gradHidden.device()
        );

        gradInputFlat = linear1_.backward(gradHidden);
    }

    // ========================================================
    // Restore input-gradient shape
    // ========================================================

    {
        ScopedProfile profile(
            profiler_,
            ProfilePhase::FFNUnflattenBackward,
            gradInputFlat.device()
        );

        gradInputFlat.reshape({ batchSize, sequenceLength, embedDim_ });
        gradInput = std::move(gradInputFlat);
    }

    return gradInput;
}

std::vector<Parameter*> FFN::parameters() {
    std::vector<Parameter*> params;

    auto p1 = linear1_.parameters();
    auto p2 = linear2_.parameters();

    params.insert(params.end(), p1.begin(), p1.end());
    params.insert(params.end(), p2.begin(), p2.end());

    return params;
}

void FFN::train() {
    training_ = true;
    dropout_.train();
}

void FFN::eval() {
    training_ = false;
    dropout_.eval();
}

bool FFN::isTraining() const {
    return training_;
}