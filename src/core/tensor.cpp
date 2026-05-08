#include "core/tensor.h"

#include <numeric>
#include <stdexcept>
#include <sstream>

Tensor::Tensor() = default;

Tensor::Tensor(const std::vector<size_t>& shape)
    : shape_(shape) {
    computeStrides();

    size_t totalSize = 1;
    for (size_t dim : shape_) {
        totalSize *= dim;
    }

    data_.resize(totalSize, 0.0f);
}

Tensor::Tensor(const std::vector<size_t>& shape, float fillValue)
    : shape_(shape) {
    computeStrides();

    size_t totalSize = 1;
    for (size_t dim : shape_) {
        totalSize *= dim;
    }

    data_.resize(totalSize, fillValue);
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

float& Tensor::at(const std::vector<size_t>& indices) {
    return data_[computeFlatIndex(indices)];
}

const float& Tensor::at(const std::vector<size_t>& indices) const {
    return data_[computeFlatIndex(indices)];
}

float& Tensor::operator[](size_t index) {
    if (index >= data_.size()) {
        throw std::out_of_range("Tensor flat index out of bounds.");
    }

    return data_[index];
}

const float& Tensor::operator[](size_t index) const {
    if (index >= data_.size()) {
        throw std::out_of_range("Tensor flat index out of bounds.");
    }

    return data_[index];
}

std::vector<float>& Tensor::data() {
    return data_;
}

const std::vector<float>& Tensor::data() const {
    return data_;
}

void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Tensor::reshape(const std::vector<size_t>& newShape) {
    size_t newSize = 1;

    for (size_t dim : newShape) {
        newSize *= dim;
    }

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