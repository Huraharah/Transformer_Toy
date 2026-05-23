#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

struct TrainingHistory {
    std::vector<float> trainLosses;
    std::vector<float> validationLosses;

    void addTrainLoss(float loss) {
        trainLosses.push_back(loss);
    }

    void addValidationLoss(float loss) {
        validationLosses.push_back(loss);
    }

    float latestTrainLoss() const {
        if (trainLosses.empty()) {
            throw std::runtime_error("No training losses recorded.");
        }
        return trainLosses.back();
    }

    float latestValidationLoss() const {
        if (validationLosses.empty()) {
            throw std::runtime_error("No validation losses recorded.");
        }
        return validationLosses.back();
    }

    void saveCsv(const std::string& path) const {
        std::ofstream file(path);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open history CSV file: " + path);
        }

        file << "step,train_loss,validation_loss\n";

        size_t maxSize = trainLosses.size();
        if (validationLosses.size() > maxSize) {
            maxSize = validationLosses.size();
        }

        for (size_t i = 0; i < maxSize; ++i) {
            file << i << ",";

            if (i < trainLosses.size()) {
                file << trainLosses[i];
            }

            file << ",";

            if (i < validationLosses.size()) {
                file << validationLosses[i];
            }

            file << "\n";
        }
    }

    void clear() {
        trainLosses.clear();
        validationLosses.clear();
    }
};
