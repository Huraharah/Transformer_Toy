#include "core/random.h"

Random::Random(uint32_t seed)
    : generator_(seed) {
}

void Random::setSeed(uint32_t seed) {
    generator_.seed(seed);
}

float Random::uniform(float minValue, float maxValue) {
    std::uniform_real_distribution<float> dist(minValue, maxValue);
    return dist(generator_);
}

float Random::normal(float mean, float stddev) {
    std::normal_distribution<float> dist(mean, stddev);
    return dist(generator_);
}

int Random::randint(int minValue, int maxValue) {
    std::uniform_int_distribution<int> dist(minValue, maxValue);
    return dist(generator_);
}