#include "training/adam_optimizer.h"
#include "kernels/optimizer_kernels.cuh"

#include <cmath>
#include <stdexcept>

AdamOptimizer::AdamOptimizer(
    float learningRate_,
    float beta1_,
    float beta2_,
    float epsilon_,
    float weightDecay_
)
    : learningRate(learningRate_),
    beta1(beta1_),
    beta2(beta2_),
    epsilon(epsilon_),
    weightDecay(weightDecay_),
    timestep(0) {

    if (learningRate <= 0.0f) {
        throw std::invalid_argument("AdamOptimizer learning rate must be > 0.");
    }

    if (beta1 <= 0.0f || beta1 >= 1.0f) {
        throw std::invalid_argument("AdamOptimizer beta1 must be in (0, 1).");
    }

    if (beta2 <= 0.0f || beta2 >= 1.0f) {
        throw std::invalid_argument("AdamOptimizer beta2 must be in (0, 1).");
    }

    if (epsilon <= 0.0f) {
        throw std::invalid_argument("AdamOptimizer epsilon must be > 0.");
    }

    if (weightDecay < 0.0f) {
        throw std::invalid_argument("AdamOptimizer weightDecay must be >= 0.");
    }
}

void AdamOptimizer::step(std::vector<Parameter*>& parameters) {
    ++timestep;

    for (Parameter* param : parameters) {
        if (!param || !param->requires_grad) {
            continue;
        }

        param->validate();

        if (m.find(param) == m.end()) {
            m[param] = Tensor(param->value.shape(), 0.0f);
            v[param] = Tensor(param->value.shape(), 0.0f);
        }

        Tensor& mt = m[param];
        Tensor& vt = v[param];

        if (param->value.device() == Device::CUDA) {
            param->value.toCUDA();
            param->grad.toCUDA();
            mt.toCUDA();
            vt.toCUDA();

            launchAdamUpdateKernel(
                param->value.deviceData(),
                param->grad.deviceData(),
                mt.deviceData(),
                vt.deviceData(),
                param->value.size(),
                learningRate,
                beta1,
                beta2,
                epsilon,
                weightDecay,
                timestep
            );

            continue;
        }

        float beta1Correction = 1.0f - std::pow(beta1, timestep);
        float beta2Correction = 1.0f - std::pow(beta2, timestep);

        for (size_t i = 0; i < param->value.size(); ++i) {
            float grad = param->grad[i];

            if (weightDecay != 0.0f) {
                grad += weightDecay * param->value[i];
            }

            mt[i] = beta1 * mt[i] + (1.0f - beta1) * grad;
            vt[i] = beta2 * vt[i] + (1.0f - beta2) * grad * grad;

            float mHat = mt[i] / beta1Correction;
            float vHat = vt[i] / beta2Correction;

            param->value[i] -= learningRate * mHat / (std::sqrt(vHat) + epsilon);
        }
    }
}

float AdamOptimizer::getLearningRate() const {
	return learningRate;
}

void AdamOptimizer::setLearningRate(float learningRate_) {
    if (learningRate_ <= 0.0f) {
        throw std::invalid_argument("Learning rate must be > 0.");
    }

    learningRate = learningRate_;
}