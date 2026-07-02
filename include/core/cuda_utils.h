#pragma once
#include "core/tensor.h"

#include <cuda_runtime.h>

void checkCuda(
    cudaError_t result,
    const char* operation,
    const char* file,
    int line
);

bool isCudaAvailable();
Device resolveDevice(Device requestedDevice);

void cudaSync();

#define CUDA_CHECK(val) \
    checkCuda((val), #val, __FILE__, __LINE__)