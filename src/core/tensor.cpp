#include "core/tensor.h"
#include "core/cuda_utils.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <cstring>

Tensor::Tensor() = default;

Tensor::Tensor(const std::vector<size_t>& shape)
    : shape_(shape) {
    computeStrides();
    data_.resize(computeTotalSize(shape_), 0.0f);
}

Tensor::Tensor(const std::vector<size_t>& shape, float fillValue)
    : shape_(shape) {
    computeStrides();
    data_.resize(computeTotalSize(shape_), fillValue);
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_),
    strides_(other.strides_),
    data_(other.data_),
    deviceData_(nullptr),
    deviceAllocated_(false),
    deviceDirty_(false),
    hostDirty_(false) {

    if (other.deviceAllocated_) {
        allocateDevice();

        if (other.hostDirty_) {
            CUDA_CHECK(
                cudaMemcpy(
                    deviceData_,
                    other.deviceData_,
                    sizeof(float) * other.size(),
                    cudaMemcpyDeviceToDevice
                )
            );

            hostDirty_ = true;
        }
        else {
            CUDA_CHECK(
                cudaMemcpy(
                    deviceData_,
                    data_.data(),
                    sizeof(float) * size(),
                    cudaMemcpyHostToDevice
                )
            );
        }
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;
    }

    freeDevice();

    shape_ = other.shape_;
    strides_ = other.strides_;
    data_ = other.data_;

    deviceData_ = nullptr;
    deviceAllocated_ = false;
    deviceDirty_ = false;
    hostDirty_ = false;

    if (other.deviceAllocated_) {
        allocateDevice();

        if (other.hostDirty_) {
            CUDA_CHECK(
                cudaMemcpy(
                    deviceData_,
                    other.deviceData_,
                    sizeof(float) * other.size(),
                    cudaMemcpyDeviceToDevice
                )
            );

            hostDirty_ = true;
        }
        else {
            CUDA_CHECK(
                cudaMemcpy(
                    deviceData_,
                    data_.data(),
                    sizeof(float) * size(),
                    cudaMemcpyHostToDevice
                )
            );
        }
    }

    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
    strides_(std::move(other.strides_)),
    data_(std::move(other.data_)),
    deviceData_(other.deviceData_),
    deviceAllocated_(other.deviceAllocated_),
    deviceDirty_(other.deviceDirty_),
    hostDirty_(other.hostDirty_) {

    other.deviceData_ = nullptr;
    other.deviceAllocated_ = false;
    other.deviceDirty_ = false;
    other.hostDirty_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    freeDevice();

    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    data_ = std::move(other.data_);

    deviceData_ = other.deviceData_;
    deviceAllocated_ = other.deviceAllocated_;
    deviceDirty_ = other.deviceDirty_;
    hostDirty_ = other.hostDirty_;

    other.deviceData_ = nullptr;
    other.deviceAllocated_ = false;
    other.deviceDirty_ = false;
    other.hostDirty_ = false;

    return *this;
}

Tensor::~Tensor() {
    freeDevice();
}

size_t Tensor::computeTotalSize(const std::vector<size_t>& shape) const {
    size_t totalSize = 1;

    for (size_t dim : shape) {
        totalSize *= dim;
    }

    return totalSize;
}

void Tensor::computeStrides() {
    strides_.resize(shape_.size());

    if (shape_.empty()) {
        return;
    }

    strides_[shape_.size() - 1] = 1;

    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

size_t Tensor::computeFlatIndex(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Index rank does not match tensor rank.");
    }

    size_t flatIndex = 0;

    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Tensor index out of bounds.");
        }

        flatIndex += indices[i] * strides_[i];
    }

    return flatIndex;
}

void Tensor::allocateDevice() {
    if (deviceAllocated_) {
        return;
    }

    if (size() == 0) {
        return;
    }

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(&deviceData_),
            sizeof(float) * size()
        )
    );

    deviceAllocated_ = true;
}

void Tensor::freeDevice() {
    if (deviceAllocated_ && deviceData_) {
        cudaFree(deviceData_);
    }

    deviceData_ = nullptr;
    deviceAllocated_ = false;
    deviceDirty_ = false;
    hostDirty_ = false;
}

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

const std::vector<size_t>& Tensor::strides() const {
    return strides_;
}

size_t Tensor::rank() const {
    return shape_.size();
}

size_t Tensor::size() const {
    return data_.size();
}

bool Tensor::empty() const {
    return data_.empty();
}

Device Tensor::device() const {
    return deviceAllocated_ ? Device::CUDA : Device::CPU;
}

bool Tensor::hasDeviceData() const {
    return deviceAllocated_;
}

void Tensor::toCUDA() {
    allocateDevice();

    if (size() == 0) {
        return;
    }

    if (!hostDirty_) {
        CUDA_CHECK(
            cudaMemcpy(
                deviceData_,
                data_.data(),
                sizeof(float) * size(),
                cudaMemcpyHostToDevice
            )
        );
    }

    deviceDirty_ = false;
}

void Tensor::toCPU() {
    if (!deviceAllocated_ || size() == 0) {
        return;
    }

    if (hostDirty_) {
        CUDA_CHECK(
            cudaMemcpy(
                data_.data(),
                deviceData_,
                sizeof(float) * size(),
                cudaMemcpyDeviceToHost
            )
        );
    }

    hostDirty_ = false;
}

float* Tensor::deviceData() {
    toCUDA();
    hostDirty_ = true;
    deviceDirty_ = false;
    return deviceData_;
}

const float* Tensor::deviceData() const {
    if (!deviceAllocated_) {
        throw std::runtime_error("Tensor has no CUDA device data.");
    }

    return deviceData_;
}

float& Tensor::at(const std::vector<size_t>& indices) {
    toCPU();
    deviceDirty_ = true;
    return data_[computeFlatIndex(indices)];
}

const float& Tensor::at(const std::vector<size_t>& indices) const {
    return data_[computeFlatIndex(indices)];
}

float& Tensor::operator[](size_t index) {
    toCPU();

    if (index >= data_.size()) {
        throw std::out_of_range("Tensor flat index out of bounds.");
    }

    deviceDirty_ = true;
    return data_[index];
}

const float& Tensor::operator[](size_t index) const {
    if (index >= data_.size()) {
        throw std::out_of_range("Tensor flat index out of bounds.");
    }

    return data_[index];
}

std::vector<float>& Tensor::data() {
    toCPU();
    deviceDirty_ = true;
    return data_;
}

const std::vector<float>& Tensor::data() const {
    return data_;
}

void Tensor::fill(float value) {
    toCPU();
    std::fill(data_.begin(), data_.end(), value);
    deviceDirty_ = true;
}

void Tensor::reshape(const std::vector<size_t>& newShape) {
    size_t newSize = computeTotalSize(newShape);

    if (newSize != data_.size()) {
        throw std::invalid_argument("Cannot reshape tensor: size mismatch.");
    }

    shape_ = newShape;
    computeStrides();
}

std::string Tensor::shapeString() const {
    std::ostringstream oss;
    oss << "[";

    for (size_t i = 0; i < shape_.size(); ++i) {
        oss << shape_[i];

        if (i + 1 < shape_.size()) {
            oss << ", ";
        }
    }

    oss << "]";
    return oss.str();
}