#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


void checkCuda(
    cudaError_t result,
    const char* operation,
    const char* file,
    int line
);

bool isCudaAvailable();

Device resolveDevice(
    Device requestedDevice
);

void cudaSync();

void setCudaSynchronizationEnabled(
    bool enabled
);

bool isCudaSynchronizationEnabled();

void cudaSyncIfEnabled();


#define CUDA_CHECK(value) \
    checkCuda(             \
        (value),           \
        #value,            \
        __FILE__,          \
        __LINE__           \
    )