#pragma once

#include <vector>
#include <cstddef>
#include <initializer_list>
#include <string>

class Tensor {
private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    std::vector<float> data_;

    void computeStrides();
    size_t computeFlatIndex(const std::vector<size_t>& indices) const;

public:
    Tensor();
    explicit Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, float fillValue);

    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& strides() const;

    size_t rank() const;
    size_t size() const;
    bool empty() const;

    float& at(const std::vector<size_t>& indices);
    const float& at(const std::vector<size_t>& indices) const;

    float& operator[](size_t index);
    const float& operator[](size_t index) const;

    std::vector<float>& data();
    const std::vector<float>& data() const;

    void fill(float value);
    void reshape(const std::vector<size_t>& newShape);

    std::string shapeString() const;
};