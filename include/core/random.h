#pragma once

#include <random>
#include <cstdint>

class Random {
private:
    std::mt19937 generator_;

public:
    explicit Random(uint32_t seed = 42);

    void setSeed(uint32_t seed);

    float uniform(float minValue, float maxValue);
    float normal(float mean, float stddev);

    int randint(int minValue, int maxValue);
};