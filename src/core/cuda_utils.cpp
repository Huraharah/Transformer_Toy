#include "core/cuda_utils.h"

#include <atomic>
#include <sstream>
#include <stdexcept>


namespace {

    std::atomic<bool> cudaSynchronizationEnabled{
        false
    };

}


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
            << "Message: "
            << cudaGetErrorString(result);

        throw std::runtime_error(
            oss.str()
        );
    }
}


bool isCudaAvailable() {
    int count = 0;

    const cudaError_t error =
        cudaGetDeviceCount(&count);

    if (error != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    return count > 0;
}


Device resolveDevice(
    Device requestedDevice
) {
    if (requestedDevice == Device::AUTO) {
        return isCudaAvailable()
            ? Device::CUDA
            : Device::CPU;
    }

    if (
        requestedDevice == Device::CUDA &&
        !isCudaAvailable()
        ) {
        return Device::CPU;
    }

    return requestedDevice;
}


void cudaSync() {
    CUDA_CHECK(
        cudaDeviceSynchronize()
    );
}


void setCudaSynchronizationEnabled(
    bool enabled
) {
    cudaSynchronizationEnabled.store(
        enabled,
        std::memory_order_relaxed
    );
}


bool isCudaSynchronizationEnabled() {
    return cudaSynchronizationEnabled.load(
        std::memory_order_relaxed
    );
}


void cudaSyncIfEnabled() {
    if (isCudaSynchronizationEnabled()) {
        cudaSync();
    }
}