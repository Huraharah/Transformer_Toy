#pragma once

#include <cstddef>

constexpr int THREADS_PER_BLOCK = 256;
constexpr int MAX_THREADS_PER_BLOCK = 1024;

inline int getBlockCount(size_t size, int threadsPerBlock = THREADS_PER_BLOCK) {
    return static_cast<int>((size + threadsPerBlock - 1) / threadsPerBlock);
}