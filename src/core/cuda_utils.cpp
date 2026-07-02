#include "core/cuda_utils.h"

#include <stdexcept>
#include <sstream>

void checkCuda(
    cudaError_t result,
    const char* operation,
    const char* file,
    int line
) {
    if (result != cudaSuccess) {
        std::ostringstream oss;

        oss
            << "CUDA ERROR\n"
            << "Operation: " << operation << "\n"
            << "File: " << file << "\n"
            << "Line: " << line << "\n"
            << "Message: " << cudaGetErrorString(result);

        throw std::runtime_error(oss.str());
    }
}

bool isCudaAvailable() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);

    if (err != cudaSuccess) {
        cudaGetLastError(); // clear CUDA error state
        return false;
    }

    return count > 0;
}

Device resolveDevice(Device requested) {
    if (requested == Device::AUTO) {
        return isCudaAvailable() ? Device::CUDA : Device::CPU;
    }

    if (requested == Device::CUDA && !isCudaAvailable()) {
        return Device::CPU; // or throw, depending on preference
    }

    return requested;
}

void cudaSync() {
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
}