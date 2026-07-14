#include "training/training_profiler.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>


void ProfileStats::addSample(
    double milliseconds
) {
    if (milliseconds < 0.0) {
        throw std::invalid_argument(
            "Profile sample cannot be negative."
        );
    }

    if (callCount == 0) {
        minimumMilliseconds = milliseconds;
        maximumMilliseconds = milliseconds;
    }
    else {
        minimumMilliseconds =
            std::min(
                minimumMilliseconds,
                milliseconds
            );

        maximumMilliseconds =
            std::max(
                maximumMilliseconds,
                milliseconds
            );
    }

    totalMilliseconds += milliseconds;
    ++callCount;
}


double ProfileStats::averageMilliseconds() const {
    if (callCount == 0) {
        return 0.0;
    }

    return
        totalMilliseconds /
        static_cast<double>(callCount);
}


TrainingProfiler::TrainingProfiler(
    bool enabled
)
    : enabled_(enabled) {
}


void TrainingProfiler::setEnabled(
    bool enabled
) {
    enabled_ = enabled;
}


bool TrainingProfiler::isEnabled() const {
    return enabled_;
}


void TrainingProfiler::start(
    ProfilePhase phase
) {
    if (!enabled_) {
        return;
    }

    ActiveTimer& timer =
        activeTimers_[phase];

    if (timer.running) {
        throw std::logic_error(
            "TrainingProfiler timer was already running: " +
            profilePhaseToString(phase)
        );
    }

    timer.startTime = Clock::now();
    timer.running = true;
}


void TrainingProfiler::stop(
    ProfilePhase phase
) {
    if (!enabled_) {
        return;
    }

    auto iterator =
        activeTimers_.find(phase);

    if (
        iterator == activeTimers_.end() ||
        !iterator->second.running
        ) {
        throw std::logic_error(
            "TrainingProfiler timer was not running: " +
            profilePhaseToString(phase)
        );
    }

    const Clock::time_point endTime =
        Clock::now();

    const double milliseconds =
        std::chrono::duration<
        double,
        std::milli
        >(
            endTime -
            iterator->second.startTime
        ).count();

    iterator->second.running = false;

    statistics_[phase].addSample(
        milliseconds
    );
}


void TrainingProfiler::addSample(
    ProfilePhase phase,
    double milliseconds
) {
    if (!enabled_) {
        return;
    }

    statistics_[phase].addSample(
        milliseconds
    );
}


void TrainingProfiler::addProcessedTokens(
    std::size_t tokens
) {
    if (!enabled_) {
        return;
    }

    processedTokens_ += tokens;
}


void TrainingProfiler::incrementProcessedSteps() {
    if (!enabled_) {
        return;
    }

    ++processedSteps_;
}


const ProfileStats& TrainingProfiler::stats(
    ProfilePhase phase
) const {
    static const ProfileStats emptyStats{};

    const auto iterator =
        statistics_.find(phase);

    if (iterator == statistics_.end()) {
        return emptyStats;
    }

    return iterator->second;
}


double TrainingProfiler::totalTrainingMilliseconds() const {
    return stats(
        ProfilePhase::TotalTraining
    ).totalMilliseconds;
}


double TrainingProfiler::stepsPerSecond() const {
    const double milliseconds =
        totalTrainingMilliseconds();

    if (milliseconds <= 0.0) {
        return 0.0;
    }

    return
        static_cast<double>(processedSteps_) /
        (milliseconds / 1000.0);
}


double TrainingProfiler::tokensPerSecond() const {
    const double milliseconds =
        totalTrainingMilliseconds();

    if (milliseconds <= 0.0) {
        return 0.0;
    }

    return
        static_cast<double>(processedTokens_) /
        (milliseconds / 1000.0);
}


void TrainingProfiler::reset() {
    statistics_.clear();
    activeTimers_.clear();

    processedTokens_ = 0;
    processedSteps_ = 0;
}


void TrainingProfiler::printSummary(
    std::ostream& output
) const {
    const ProfilePhase phases[] = {
        ProfilePhase::BatchTransfer,
        ProfilePhase::ModelForward,
        ProfilePhase::LossForward,
        ProfilePhase::LossBackward,
        ProfilePhase::ModelBackward,
        ProfilePhase::OptimizerStep,
        ProfilePhase::ZeroGrad,
        ProfilePhase::TrainingStep,
        ProfilePhase::Validation,
        ProfilePhase::Callbacks,
        ProfilePhase::Generation,
        ProfilePhase::Checkpoint,
        ProfilePhase::Epoch,
        ProfilePhase::TotalTraining
    };

    output
        << "\n"
        << "============================================================\n"
        << "||                 Training Profile Summary               ||\n"
        << "============================================================\n";

    output
        << std::left
        << std::setw(22) << "Phase"
        << std::right
        << std::setw(10) << "Calls"
        << std::setw(14) << "Total ms"
        << std::setw(14) << "Average ms"
        << std::setw(14) << "Minimum ms"
        << std::setw(14) << "Maximum ms"
        << "\n";

    output
        << std::string(88, '-')
        << "\n";

    for (ProfilePhase phase : phases) {
        const ProfileStats& phaseStats =
            stats(phase);

        if (phaseStats.callCount == 0) {
            continue;
        }

        output
            << std::left
            << std::setw(22)
            << profilePhaseToString(phase)
            << std::right
            << std::setw(10)
            << phaseStats.callCount
            << std::setw(14)
            << std::fixed
            << std::setprecision(3)
            << phaseStats.totalMilliseconds
            << std::setw(14)
            << phaseStats.averageMilliseconds()
            << std::setw(14)
            << phaseStats.minimumMilliseconds
            << std::setw(14)
            << phaseStats.maximumMilliseconds
            << "\n";
    }

    output
        << std::string(88, '-')
        << "\n"
        << "Processed steps:  "
        << processedSteps_
        << "\n"
        << "Processed tokens: "
        << processedTokens_
        << "\n"
        << "Steps/second:     "
        << std::fixed
        << std::setprecision(3)
        << stepsPerSecond()
        << "\n"
        << "Tokens/second:    "
        << tokensPerSecond()
        << "\n"
        << "============================================================\n";
}


std::string profilePhaseToString(
    ProfilePhase phase
) {
    switch (phase) {
    case ProfilePhase::BatchTransfer:
        return "Batch transfer";

    case ProfilePhase::ModelForward:
        return "Model forward";

    case ProfilePhase::LossForward:
        return "Loss forward";

    case ProfilePhase::LossBackward:
        return "Loss backward";

    case ProfilePhase::ModelBackward:
        return "Model backward";

    case ProfilePhase::OptimizerStep:
        return "Optimizer step";

    case ProfilePhase::ZeroGrad:
        return "Zero gradient";

    case ProfilePhase::TrainingStep:
        return "Training step";

    case ProfilePhase::Validation:
        return "Validation";

    case ProfilePhase::Callbacks:
        return "Callbacks";

    case ProfilePhase::Generation:
        return "Generation";

    case ProfilePhase::Checkpoint:
        return "Checkpoint";

    case ProfilePhase::Epoch:
        return "Epoch";

    case ProfilePhase::TotalTraining:
        return "Total training";

    default:
        return "Unknown";
    }
}