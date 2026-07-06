#include "core/layer_utils.h"
#include "kernels/tensor_ops_kernels.cuh"

#include <iostream>

Tensor LayerUtils::flatten3DTo2D(const Tensor& input) {
    if (input.rank() != 3) {
        throw std::invalid_argument("flatten3DTo2D expects input shape [batch, sequence, features].");
    }

    size_t batchSize = input.shape()[0];
    size_t sequenceLength = input.shape()[1];
    size_t featureDim = input.shape()[2];

    Tensor output({ batchSize * sequenceLength, featureDim }, 0.0f);

    if (input.device() == Device::CUDA) {
        Tensor& mutableInput = const_cast<Tensor&>(input);

        mutableInput.toCUDA();
        output.toCUDA();

        launchFlatten3DTo2D(
            mutableInput.deviceData(),
            output.deviceData(),
            batchSize,
            sequenceLength,
            featureDim
        );

        return output;
    }

    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t t = 0; t < sequenceLength; ++t) {
            for (size_t f = 0; f < featureDim; ++f) {
                output.at({ b * sequenceLength + t, f }) =
                    input.at({ b, t, f });
            }
        }
    }

    return output;
}

Tensor LayerUtils::unflatten2DTo3D(
    const Tensor& input,
    size_t batchSize,
    size_t sequenceLength
) {
    if (input.rank() != 2) {
        throw std::invalid_argument("unflatten2DTo3D expects input shape [batch * sequence, features].");
    }

    size_t featureDim = input.shape()[1];

    Tensor output({ batchSize, sequenceLength, featureDim }, 0.0f);

    if (input.device() == Device::CUDA) {
        Tensor& mutableInput = const_cast<Tensor&>(input);

        mutableInput.toCUDA();
        output.toCUDA();

        launchUnflatten2DTo3D(
            mutableInput.deviceData(),
            output.deviceData(),
            batchSize,
            sequenceLength,
            featureDim
        );

        return output;
    }

    for (size_t b = 0; b < batchSize; ++b) {
        for (size_t t = 0; t < sequenceLength; ++t) {
            for (size_t f = 0; f < featureDim; ++f) {
                output.at({ b, t, f }) =
                    input.at({ b * sequenceLength + t, f });
            }
        }
    }

    return output;
}