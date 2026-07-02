#pragma once

#include <vector>
#include <cstddef>
#include <initializer_list>
#include <string>

enum class Device {
    CPU,
    CUDA,
    AUTO
};

class Tensor {
private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    std::vector<float> data_;

    float* deviceData_ = nullptr;
    bool deviceAllocated_ = false;
    bool deviceDirty_ = false;
    bool hostDirty_ = false;

    void computeStrides();
    size_t computeFlatIndex(const std::vector<size_t>& indices) const;
    size_t computeTotalSize(const std::vector<size_t>& shape) const;

    void allocateDevice();
    void freeDevice();

public:
    Tensor();
    explicit Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, float fillValue);

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();

    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& strides() const;

    size_t rank() const;
    size_t size() const;
    bool empty() const;

    Device device() const;
    bool hasDeviceData() const;

    void toCUDA();
    void toCPU();

    float* deviceData();
    const float* deviceData() const;

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