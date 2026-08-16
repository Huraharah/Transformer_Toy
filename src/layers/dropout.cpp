#include "layers/dropout.h"
#include "kernels/dropout_kernels.cuh"

Dropout::Dropout(float probability, Random& rng)
	: probability_(probability),
    scale_(1.0f),
    hasMask_(false),
	rng_(rng),
	training_(true) {
    if (probability < 0.0f || probability >= 1.0f) {
        throw std::invalid_argument(
            "Dropout probability must be in [0.0f, 1.0f)"
        );
    };

	scale_ = 1.0f / (1.0f - probability_);
	
}

Tensor Dropout::forward(const Tensor& input) {
    if (!training_ || probability_ == 0.0f ) {
        hasMask_ = false;
        return input;
    }

    mask_ = Tensor(input.shape(), 0.0f);

    for (size_t index = 0; index < input.size(); ++index) {
        const bool keep = rng_.uniform(0.0f, 1.0f) >= probability_;

        if (keep) {
            mask_[index] = scale_;
        }
    }

    Tensor output(input.shape(), 0.0f);

    if (input.device() == Device::CUDA) {
        Tensor& mutableInput = const_cast<Tensor&>(input);

        mutableInput.toCUDA();
        mask_.toCUDA();
        output.toCUDA();

        launchDropoutForward(
            mutableInput.deviceData(),
            mask_.deviceData(),
            output.deviceData(),
            input.size()
        );
    }
    else {
        for (size_t index = 0; index < input.size(); ++index) {
            output[index] = input[index] * mask_[index];
        }
    }

    hasMask_ = true;

    return output;
}

Tensor Dropout::backward(const Tensor& gradOutput) {
    if (!training_ || probability_ == 0.0f) {
        return gradOutput;
    }

    if (!hasMask_) {
        throw std::runtime_error(
            "Dropout::backward called before training forward."
        );
    }

    if (gradOutput.shape() != mask_.shape()) {
        throw std::invalid_argument(
            "Dropout::backward gradient shape mismatch."
        );
    }

    Tensor gradInput(gradOutput.shape(), 0.0f);

    if (gradOutput.device() == Device::CUDA) {
        Tensor& mutableGradOutput = const_cast<Tensor&>(gradOutput);

        mutableGradOutput.toCUDA();
        mask_.toCUDA();
        gradInput.toCUDA();

        launchDropoutBackward(
            mutableGradOutput.deviceData(),
            mask_.deviceData(),
            gradInput.deviceData(),
            gradOutput.size()
        );
    }
    else {
        for (size_t index = 0; index < gradOutput.size(); ++index) {
            gradInput[index] = gradOutput[index] * mask_[index];
        }
    }

    return gradInput;
}

void Dropout::eval() {
    training_ = false;
    mask_ = Tensor();
    hasMask_ = false;
}

void Dropout::train() {
    training_ = true;
    mask_ = Tensor();
    hasMask_ = false;
}
	
bool Dropout::isTraining() const {
    return training_;
}

float Dropout::probability() const {
    return probability_;
}