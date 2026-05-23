#include "training/sgd_optimizer.h"

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

void SGDOptimizer::step(std::vector<Parameter*>& parameters) {
    for (Parameter* param : parameters) {
        if (!param || !param->requires_grad) {
            continue;
        }

        param->validate();

        for (size_t i = 0; i < param->value.size(); ++i) {
            float grad = param->grad[i];

            if (weightDecay != 0.0f) {
                grad += weightDecay * param->value[i];
            }

            param->value[i] -= learningRate * grad;
        }
    }
}