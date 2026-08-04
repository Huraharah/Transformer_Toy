#include "training/checkpoint.h"

#include <fstream>
#include <stdexcept>
#include <cstdint>

static void writeString(std::ofstream& out, const std::string& value) {
    uint64_t size = static_cast<uint64_t>(value.size());
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(value.data(), size);
}

static std::string readString(
    std::ifstream& in,
    uint64_t maxSize = 1024 * 1024
) {
    uint64_t size = 0;

    if (!in.read(
        reinterpret_cast<char*>(&size),
        sizeof(size)
    )) {
        throw std::runtime_error(
            "Checkpoint ended while reading string size."
        );
    }

    if (size > maxSize) {
        throw std::runtime_error(
            "Checkpoint contains an invalid string size."
        );
    }

    std::string value(
        static_cast<size_t>(size),
        '\0'
    );

    if (
        size > 0 &&
        !in.read(
            value.data(),
            static_cast<std::streamsize>(size)
        )
        ) {
        throw std::runtime_error(
            "Checkpoint ended while reading string data."
        );
    }

    return value;
}

static void writeTensor(std::ofstream& out, const Tensor& tensor) {
    uint64_t rank = static_cast<uint64_t>(tensor.rank());
    out.write(reinterpret_cast<const char*>(&rank), sizeof(rank));

    for (size_t dim : tensor.shape()) {
        uint64_t d = static_cast<uint64_t>(dim);
        out.write(reinterpret_cast<const char*>(&d), sizeof(d));
    }

    uint64_t size = static_cast<uint64_t>(tensor.size());
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));

    out.write(
        reinterpret_cast<const char*>(tensor.data().data()),
        sizeof(float) * size
    );
}

static Tensor readTensor(std::ifstream& in) {
    constexpr uint64_t MAX_RANK = 16;
    constexpr uint64_t MAX_ELEMENTS =
        1ULL << 34;  // Adjust to an appropriate project limit.

    uint64_t rank = 0;

    if (!in.read(
        reinterpret_cast<char*>(&rank),
        sizeof(rank)
    )) {
        throw std::runtime_error(
            "Checkpoint ended while reading tensor rank."
        );
    }

    if (rank > MAX_RANK) {
        throw std::runtime_error(
            "Checkpoint contains an invalid tensor rank."
        );
    }

    std::vector<size_t> shape(
        static_cast<size_t>(rank)
    );

    uint64_t computedSize = 1;

    for (uint64_t index = 0; index < rank; ++index) {
        uint64_t dimension = 0;

        if (!in.read(
            reinterpret_cast<char*>(&dimension),
            sizeof(dimension)
        )) {
            throw std::runtime_error(
                "Checkpoint ended while reading tensor shape."
            );
        }

        if (dimension == 0) {
            throw std::runtime_error(
                "Checkpoint contains a zero tensor dimension."
            );
        }

        if (
            computedSize >
            MAX_ELEMENTS / dimension
            ) {
            throw std::runtime_error(
                "Checkpoint tensor size exceeds the allowed limit."
            );
        }

        computedSize *= dimension;

        shape[static_cast<size_t>(index)] =
            static_cast<size_t>(dimension);
    }

    uint64_t storedSize = 0;

    if (!in.read(
        reinterpret_cast<char*>(&storedSize),
        sizeof(storedSize)
    )) {
        throw std::runtime_error(
            "Checkpoint ended while reading tensor size."
        );
    }

    if (
        storedSize != computedSize ||
        storedSize > MAX_ELEMENTS
        ) {
        throw std::runtime_error(
            "Checkpoint tensor size is invalid."
        );
    }

    Tensor tensor(shape);

    const std::streamsize byteCount =
        static_cast<std::streamsize>(
            sizeof(float) * storedSize
            );

    if (
        storedSize > 0 &&
        !in.read(
            reinterpret_cast<char*>(
                tensor.data().data()
                ),
            byteCount
        )
        ) {
        throw std::runtime_error(
            "Checkpoint ended while reading tensor data."
        );
    }

    return tensor;
}

void Checkpoint::save(
    const std::string& path,
    const std::vector<Parameter*>& parameters,
    const CheckpointMetadata& metadata,
    const TrainingHistory& history
) {
    std::ofstream out(path, std::ios::binary);

    if (!out.is_open()) {
        throw std::runtime_error("Failed to open checkpoint for writing: " + path);
    }

    writeString(out, "TRANSFORMER_TOY_CHECKPOINT_V1");

    out.write(reinterpret_cast<const char*>(&metadata.epoch), sizeof(metadata.epoch));
    out.write(reinterpret_cast<const char*>(&metadata.globalStep), sizeof(metadata.globalStep));
    writeString(out, metadata.runName);

    uint64_t paramCount = static_cast<uint64_t>(parameters.size());
    out.write(reinterpret_cast<const char*>(&paramCount), sizeof(paramCount));

    for (Parameter* param : parameters) {
        if (!param) {
            throw std::runtime_error("Cannot save null Parameter pointer.");
        }

        writeString(out, param->name);
        out.write(reinterpret_cast<const char*>(&param->requires_grad), sizeof(param->requires_grad));
        writeTensor(out, param->value);
        writeTensor(out, param->grad);
    }

    uint64_t trainCount = static_cast<uint64_t>(history.trainLosses.size());
    out.write(reinterpret_cast<const char*>(&trainCount), sizeof(trainCount));

    for (float loss : history.trainLosses) {
        out.write(reinterpret_cast<const char*>(&loss), sizeof(loss));
    }

    uint64_t valCount = static_cast<uint64_t>(history.validationLosses.size());
    out.write(reinterpret_cast<const char*>(&valCount), sizeof(valCount));

    for (float loss : history.validationLosses) {
        out.write(reinterpret_cast<const char*>(&loss), sizeof(loss));
    }
}

void Checkpoint::load(
    const std::string& path,
    std::vector<Parameter*>& parameters,
    CheckpointMetadata& metadata,
    TrainingHistory& history
) {
    std::ifstream in(path, std::ios::binary);

    if (!in.is_open()) {
        throw std::runtime_error("Failed to open checkpoint for reading: " + path);
    }

    std::string magic = readString(in);

    if (magic != "TRANSFORMER_TOY_CHECKPOINT_V1") {
        throw std::runtime_error("Invalid checkpoint file format.");
    }

    in.read(reinterpret_cast<char*>(&metadata.epoch), sizeof(metadata.epoch));
    in.read(reinterpret_cast<char*>(&metadata.globalStep), sizeof(metadata.globalStep));
    metadata.runName = readString(in);

    uint64_t paramCount = 0;
    in.read(reinterpret_cast<char*>(&paramCount), sizeof(paramCount));

    if (paramCount != parameters.size()) {
        throw std::runtime_error("Checkpoint parameter count does not match model parameter count.");
    }

    for (uint64_t i = 0; i < paramCount; ++i) {
        Parameter* param = parameters[i];

        if (!param) {
            throw std::runtime_error("Cannot load into null Parameter pointer.");
        }

        std::string savedName = readString(in);
        bool savedRequiresGrad = true;

        in.read(reinterpret_cast<char*>(&savedRequiresGrad), sizeof(savedRequiresGrad));

        Tensor savedValue = readTensor(in);
        Tensor savedGrad = readTensor(in);

        if (savedValue.size() != param->value.size()) {
            throw std::runtime_error("Checkpoint parameter size mismatch for: " + savedName);
        }

        param->name = savedName;
        param->requires_grad = savedRequiresGrad;
        param->value = savedValue;
        param->grad = savedGrad;
    }

    history.clear();

    uint64_t trainCount = 0;
    in.read(reinterpret_cast<char*>(&trainCount), sizeof(trainCount));

    for (uint64_t i = 0; i < trainCount; ++i) {
        float loss = 0.0f;
        in.read(reinterpret_cast<char*>(&loss), sizeof(loss));
        history.trainLosses.push_back(loss);
    }

    uint64_t valCount = 0;
    in.read(reinterpret_cast<char*>(&valCount), sizeof(valCount));

    for (uint64_t i = 0; i < valCount; ++i) {
        float loss = 0.0f;
        in.read(reinterpret_cast<char*>(&loss), sizeof(loss));
        history.validationLosses.push_back(loss);
    }
}