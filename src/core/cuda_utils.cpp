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

void cudaSync() {
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
}