#pragma once

#include <chrono>
#include <cstddef>
#include <iosfwd>
#include <string>
#include <unordered_map>


enum class ProfilePhase {
    BatchTransfer,
    ModelForward,
    LossForward,
    LossBackward,
    ModelBackward,
    OptimizerStep,
    ZeroGrad,
    TrainingStep,
    Validation,
    Callbacks,
    Generation,
    Checkpoint,
    Epoch,
    TotalTraining
};


struct ProfileStats {
    double totalMilliseconds = 0.0;
    double minimumMilliseconds = 0.0;
    double maximumMilliseconds = 0.0;

    std::size_t callCount = 0;

    void addSample(double milliseconds);

    double averageMilliseconds() const;
};


class TrainingProfiler {
private:
    using Clock =
        std::chrono::steady_clock;

    struct ActiveTimer {
        Clock::time_point startTime;
        bool running = false;
    };

    std::unordered_map<
        ProfilePhase,
        ProfileStats
    > statistics_;

    std::unordered_map<
        ProfilePhase,
        ActiveTimer
    > activeTimers_;

    std::size_t processedTokens_ = 0;
    std::size_t processedSteps_ = 0;

    bool enabled_ = true;

public:
    explicit TrainingProfiler(
        bool enabled = true
    );

    void setEnabled(bool enabled);
    bool isEnabled() const;

    void start(ProfilePhase phase);
    void stop(ProfilePhase phase);

    void addSample(
        ProfilePhase phase,
        double milliseconds
    );

    void addProcessedTokens(
        std::size_t tokens
    );

    void incrementProcessedSteps();

    const ProfileStats& stats(
        ProfilePhase phase
    ) const;

    double totalTrainingMilliseconds() const;
    double stepsPerSecond() const;
    double tokensPerSecond() const;

    void reset();

    void printSummary(
        std::ostream& output
    ) const;
};


std::string profilePhaseToString(
    ProfilePhase phase
);