#include "training/cross_entropy_loss.h"

#include <cmath>
#include <stdexcept>
#include <limits>

float CrossEntropyLoss::forward(
    const Tensor& logits,
    const Tensor& targets
) {
    if (logits.rank() != 2) {
        throw std::invalid_argument("CrossEntropyLoss expects logits with shape [batch, classes].");
    }

    if (targets.rank() != 1) {
        throw std::invalid_argument("CrossEntropyLoss expects targets with shape [batch].");
    }

    size_t batchSize = logits.shape()[0];
    size_t numClasses = logits.shape()[1];

    if (targets.shape()[0] != batchSize) {
        throw std::invalid_argument("CrossEntropyLoss target batch size does not match logits batch size.");
    }

    cachedGrad = Tensor(logits.shape(), 0.0f);

    if (logits.device() == Device::CUDA) {
        cachedGrad.toCUDA();

        Tensor losses({ batchSize }, 0.0f);
        losses.toCUDA();

        Tensor& mutableLogits = const_cast<Tensor&>(logits);
        Tensor& mutableTargets = const_cast<Tensor&>(targets);

        mutableLogits.toCUDA();
        mutableTargets.toCUDA();

        launchCrossEntropyForwardBackwardKernel(
            mutableLogits.deviceData(),
            mutableTargets.deviceData(),
            cachedGrad.deviceData(),
            losses.deviceData(),
            batchSize,
            numClasses
        );

        losses.toCPU();

        float totalLoss = 0.0f;
        for (size_t i = 0; i < losses.size(); ++i) {
            totalLoss += losses[i];
        }

        return totalLoss / static_cast<float>(batchSize);
    }

    else {

        float totalLoss = 0.0f;

        for (size_t b = 0; b < batchSize; ++b) {
            int targetClass = static_cast<int>(targets[b]);

            if (targetClass < 0 || static_cast<size_t>(targetClass) >= numClasses) {
                throw std::out_of_range("CrossEntropyLoss target class index out of range.");
            }

            float maxLogit = -std::numeric_limits<float>::infinity();

            for (size_t c = 0; c < numClasses; ++c) {
                float val = logits[b * numClasses + c];
                if (val > maxLogit) {
                    maxLogit = val;
                }
            }

            float sumExp = 0.0f;

            for (size_t c = 0; c < numClasses; ++c) {
                sumExp += std::exp(logits[b * numClasses + c] - maxLogit);
            }

            float logSumExp = maxLogit + std::log(sumExp);
            totalLoss += -logits[b * numClasses + targetClass] + logSumExp;

            for (size_t c = 0; c < numClasses; ++c) {
                float softmax = std::exp(logits[b * numClasses + c] - logSumExp);
                cachedGrad[b * numClasses + c] = softmax;
            }

            cachedGrad[b * numClasses + targetClass] -= 1.0f;
        }

        float invBatch = 1.0f / static_cast<float>(batchSize);

        for (size_t i = 0; i < cachedGrad.size(); ++i) {
            cachedGrad[i] *= invBatch;
        }

        return totalLoss * invBatch;
    }
}

Tensor CrossEntropyLoss::backward() const {
    return cachedGrad;
}