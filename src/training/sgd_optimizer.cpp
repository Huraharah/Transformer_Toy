#include "training/sgd_optimizer.h"
#include "core/cuda_utils.h"
#include "kernels/sgd_kernel.cuh"

#include <stdexcept>

SGDOptimizer::SGDOptimizer(float learningRate_, float weightDecay_)
    : learningRate(learningRate_),
      weightDecay(weightDecay_) {

    if (learningRate <= 0.0f) {
        throw std::invalid_argument("SGDOptimizer learning rate must be > 0.");
    }

    if (weightDecay < 0.0f) {
        throw std::invalid_argument("SGDOptimizer weight decay must be >= 0.");
    }
}

static constexpr int THREADS_PER_BLOCK = 256;

void SGDOptimizer::step(std::vector<Parameter*>& parameters) {
    for (Parameter* param : parameters) {
        if (!param || !param->requires_grad) {
            continue;
        }

        param->validate();

        if (param->value.device() == Device::CUDA) {
            param->value.toCUDA();
            param->grad.toCUDA();

            int blocks = static_cast<int>(
                (param->value.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK
                );

            launchSGDUpdateKernel(
                param->value.deviceData(),
                param->grad.deviceData(),
                param->value.size(),
                learningRate,
                weightDecay
                );

            CUDA_CHECK(cudaGetLastError());
            cudaSync();
        }
        else {
            for (size_t i = 0; i < param->value.size(); ++i) {
                float grad = param->grad[i];

                if (weightDecay != 0.0f) {
                    grad += weightDecay * param->value[i];
                }

                param->value[i] -= learningRate * grad;
            }
        }
    }
}