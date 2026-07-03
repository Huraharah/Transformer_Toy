#include "tests/SmokeTests.h"

#include "training/training_config.h"
#include "core/tensor.h"
#include "core/cuda_utils.h"
#include "data/dataset.h"
#include "core/random.h"
#include "training/trainer.h"

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>

namespace {

    void printSection(const std::string& title) {
        std::cout << "\n==================================================\n";
        std::cout << "|| " << title << "\n";
        std::cout << "==================================================\n\n";
    }

    bool cudaAvailableForSmokeTest() {
        return isCudaAvailable();
    }

    void runTinyShakespeareSmokeTest(Device device) {
        std::cout << "[SMOKE] Tiny Shakespeare test on "
            << (device == Device::CUDA ? "CUDA" : "CPU")
            << "\n";

        TrainingConfig config;
        config.device = device;
        config.epochs = 1;
        config.batchSize = 4;
        config.logEverySteps = 1;
        config.checkpointEveryEpochs = 0;
        config.runName = device == Device::CUDA
            ? "tiny_shakespeare_cuda_smoke"
            : "tiny_shakespeare_cpu_smoke";
        config.checkpointDirectory = ".";

        constexpr size_t contextLength = 32;
        constexpr size_t batchSize = 4;

        TextDataset dataset("data/shakespeare.txt", contextLength);

        std::cout
            << "[INFO] Dataset loaded"
            << " | vocab=" << dataset.vocabSize()
            << " | context=" << dataset.contextLength()
            << " | windows=" << dataset.numWindows()
            << "\n";

        Random rng(42);

        Tensor inputs;
        Tensor targets;

        dataset.getBatch(batchSize, rng, inputs, targets);

        std::cout
            << "[INFO] Batch created"
            << " | inputs=" << inputs.shapeString()
            << " | targets=" << targets.shapeString()
            << "\n";

        TrainingBatch batch;
        batch.inputs = inputs;
        batch.targets = targets;

        std::vector<TrainingBatch> trainBatches = { batch };


        for (size_t i = 0; i < 20; ++i) {
            Tensor batchInputs;
            Tensor batchTargets;

            dataset.getBatch(batchSize, rng, batchInputs, batchTargets);

            trainBatches.push_back({ batchInputs, batchTargets });
        }

        std::cout << "[PASS] Smoke test scaffold completed for "
            << (device == Device::CUDA ? "CUDA" : "CPU")
            << "\n";
    }

}

void runTinyShakespeareSmokeTestCPU() {
    runTinyShakespeareSmokeTest(Device::CPU);
}

void runTinyShakespeareSmokeTestCUDA() {
    if (!cudaAvailableForSmokeTest()) {
        std::cout << "[SKIP] CUDA smoke test skipped: CUDA not available\n";
        return;
    }

    runTinyShakespeareSmokeTest(Device::CUDA);
}

void runSmokeTests() {
    printSection("Smoke Tests");

    runTinyShakespeareSmokeTestCPU();
    runTinyShakespeareSmokeTestCUDA();

    std::cout << "\n[PASS] Smoke test suite completed\n";
}