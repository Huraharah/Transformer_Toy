#include "training/training_profiler.h"

#include "core/cuda_utils.h"

/*
    Replace this include path with the actual header containing the
    complete TrainableModel class definition.
*/
#include "training/trainer.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>


// ============================================================
// ProfileStats
// ============================================================

void ProfileStats::addSample(
    double milliseconds
) {
    if (milliseconds < 0.0) {
        throw std::invalid_argument(
            "Profile sample cannot be negative."
        );
    }

    if (callCount == 0) {
        minimumMilliseconds =
            milliseconds;

        maximumMilliseconds =
            milliseconds;
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

    totalMilliseconds +=
        milliseconds;

    ++callCount;
}


double ProfileStats::averageMilliseconds() const {
    if (callCount == 0) {
        return 0.0;
    }

    return
        totalMilliseconds /
        static_cast<double>(
            callCount
            );
}


// ============================================================
// TrainingProfiler
// ============================================================

TrainingProfiler::TrainingProfiler(
    bool enabled
)
    : enabled_(enabled),
    synchronizeCudaPhases_(false) {
}


void TrainingProfiler::setEnabled(
    bool enabled
) {
    enabled_ = enabled;
}


bool TrainingProfiler::isEnabled() const {
    return enabled_;
}


void TrainingProfiler::setSynchronizeCudaPhases(
    bool enabled
) {
    synchronizeCudaPhases_ =
        enabled;
}


bool TrainingProfiler::synchronizeCudaPhases() const {
    return synchronizeCudaPhases_;
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
            profilePhaseToString(
                phase
            )
        );
    }

    timer.startTime =
        Clock::now();

    timer.running =
        true;
}


void TrainingProfiler::stop(
    ProfilePhase phase
) {
    if (!enabled_) {
        return;
    }

    auto iterator =
        activeTimers_.find(
            phase
        );

    if (
        iterator ==
        activeTimers_.end() ||
        !iterator->second.running
        ) {
        throw std::logic_error(
            "TrainingProfiler timer was not running: " +
            profilePhaseToString(
                phase
            )
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

    iterator->second.running =
        false;

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

    processedTokens_ +=
        tokens;
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
        statistics_.find(
            phase
        );

    if (
        iterator ==
        statistics_.end()
        ) {
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
        static_cast<double>(
            processedSteps_
            ) /
        (
            milliseconds /
            1000.0
            );
}


double TrainingProfiler::tokensPerSecond() const {
    const double milliseconds =
        totalTrainingMilliseconds();

    if (milliseconds <= 0.0) {
        return 0.0;
    }

    return
        static_cast<double>(
            processedTokens_
            ) /
        (
            milliseconds /
            1000.0
            );
}


void TrainingProfiler::reset() {
    statistics_.clear();
    activeTimers_.clear();

    processedTokens_ = 0;
    processedSteps_ = 0;

    /*
        Do not reset enabled_ or synchronizeCudaPhases_ here.

        Those values are profiler configuration rather than recorded
        statistics.
    */
}


// ============================================================
// ScopedProfile
// ============================================================

ScopedProfile::ScopedProfile(
    TrainingProfiler* profiler,
    ProfilePhase phase,
    Device device
)
    : profiler_(profiler),
    phase_(phase),
    device_(device),
    active_(
        profiler != nullptr &&
        profiler->isEnabled()
    ) {

    if (active_) {
        profiler_->start(
            phase_
        );
    }
}


ScopedProfile::~ScopedProfile() {
    if (!active_) {
        return;
    }

    /*
        CUDA wrappers no longer synchronize.

        When synchronized profiling is enabled, wait at the end of the
        scoped phase so the recorded time includes completed GPU work.
    */
    if (
        device_ == Device::CUDA &&
        profiler_->synchronizeCudaPhases()
        ) {
        cudaSync();
    }

    profiler_->stop(
        phase_
    );
}


// ============================================================
// ScopedModelProfiler
// ============================================================

ScopedModelProfiler::ScopedModelProfiler(
    TrainableModel& model,
    TrainingProfiler* profiler
)
    : model_(model) {

    model_.setProfiler(
        profiler
    );
}


ScopedModelProfiler::~ScopedModelProfiler() {
    model_.setProfiler(
        nullptr
    );
}


// ============================================================
// Summary output
// ============================================================

void TrainingProfiler::printSummary(
    std::ostream& output
) const {
    const ProfilePhase phases[] = {
        ProfilePhase::BatchTransfer,

        ProfilePhase::ModelForward,
        ProfilePhase::TransformerEmbedding,
        ProfilePhase::TransformerBlocksForward,
        ProfilePhase::BlockNorm1Forward,
        ProfilePhase::BlockAttentionForward,
        ProfilePhase::AttentionFlattenForward,
        ProfilePhase::AttentionQProjectionForward,
        ProfilePhase::AttentionKProjectionForward,
        ProfilePhase::AttentionVProjectionForward,
        ProfilePhase::AttentionCacheForward,
        ProfilePhase::AttentionKernelForward,
        ProfilePhase::AttentionOutputProjectionForward,
        ProfilePhase::AttentionUnflattenForward,
        ProfilePhase::BlockResidual1Forward,
        ProfilePhase::BlockNorm2Forward,
        ProfilePhase::BlockFFNForward,
        ProfilePhase::FFNFlattenForward,
        ProfilePhase::FFNLinear1Forward,
        ProfilePhase::FFNGeluForward,
        ProfilePhase::FFNLinear2Forward,
        ProfilePhase::FFNUnflattenForward,
        ProfilePhase::BlockResidual2Forward,
        ProfilePhase::TransformerFinalNorm,
        ProfilePhase::TransformerOutputProjection,

        ProfilePhase::LossForward,
        ProfilePhase::LossBackward,

        ProfilePhase::ModelBackward,
        ProfilePhase::TransformerOutputBackward,
        ProfilePhase::TransformerFinalNormBackward,
        ProfilePhase::TransformerBlocksBackward,
        ProfilePhase::BlockFFNBackward,
        ProfilePhase::FFNFlattenBackward,
        ProfilePhase::FFNLinear2Backward,
        ProfilePhase::FFNGeluBackward,
        ProfilePhase::FFNLinear1Backward,
        ProfilePhase::FFNUnflattenBackward,
        ProfilePhase::BlockNorm2Backward,
        ProfilePhase::BlockResidual1Backward,
        ProfilePhase::BlockAttentionBackward,
        ProfilePhase::AttentionFlattenBackward,
        ProfilePhase::AttentionOutputProjectionBackward,
        ProfilePhase::AttentionKernelBackward,
        ProfilePhase::AttentionQProjectionBackward,
        ProfilePhase::AttentionKProjectionBackward,
        ProfilePhase::AttentionVProjectionBackward,
        ProfilePhase::AttentionGradientMergeBackward,
        ProfilePhase::AttentionUnflattenBackward,
        ProfilePhase::BlockNorm1Backward,
        ProfilePhase::BlockInputMergeBackward,
        ProfilePhase::TransformerEmbeddingBackward,

        ProfilePhase::OptimizerStep,
        ProfilePhase::ZeroGrad,

        ProfilePhase::TrainingStep,
        ProfilePhase::Validation,
        ProfilePhase::Callbacks,
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
        << std::setw(36)
        << "Phase"
        << std::right
        << std::setw(10)
        << "Calls"
        << std::setw(14)
        << "Total ms"
        << std::setw(14)
        << "Average ms"
        << std::setw(14)
        << "Minimum ms"
        << std::setw(14)
        << "Maximum ms"
        << "\n";

    output
        << std::string(
            102,
            '-'
        )
        << "\n";

    for (
        ProfilePhase phase :
    phases
        ) {
        const ProfileStats& phaseStats =
            stats(
                phase
            );

        if (
            phaseStats.callCount == 0
            ) {
            continue;
        }

        output
            << std::left
            << std::setw(36)
            << profilePhaseToString(
                phase
            )
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
        << std::string(
            102,
            '-'
        )
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


// ============================================================
// Phase names
// ============================================================

std::string profilePhaseToString(
    ProfilePhase phase
) {
    switch (phase) {
    case ProfilePhase::BatchTransfer:
        return "Batch transfer";

    case ProfilePhase::ModelForward:
        return "Model forward";

    case ProfilePhase::TransformerEmbedding:
        return "  Transformer embedding";

    case ProfilePhase::TransformerBlocksForward:
        return "  Transformer blocks forward";

    case ProfilePhase::BlockNorm1Forward:
        return "    Block norm1";

    case ProfilePhase::BlockAttentionForward:
        return "    Block attention";

    case ProfilePhase::BlockResidual1Forward:
        return "    Block residual1";

    case ProfilePhase::BlockNorm2Forward:
        return "    Block norm2";

    case ProfilePhase::BlockFFNForward:
        return "    Block FFN";

    case ProfilePhase::FFNFlattenForward:
        return "      FFN flatten";

    case ProfilePhase::FFNLinear1Forward:
        return "      FFN linear1";

    case ProfilePhase::FFNGeluForward:
        return "      FFN GELU";

    case ProfilePhase::FFNLinear2Forward:
        return "      FFN linear2";

    case ProfilePhase::FFNUnflattenForward:
        return "      FFN unflatten";

    case ProfilePhase::BlockResidual2Forward:
        return "    Block residual2";

    case ProfilePhase::TransformerFinalNorm:
        return "  Transformer final norm";

    case ProfilePhase::TransformerOutputProjection:
        return "  Transformer output projection";

    case ProfilePhase::LossForward:
        return "Loss forward";

    case ProfilePhase::LossBackward:
        return "Loss backward";

    case ProfilePhase::ModelBackward:
        return "Model backward";

    case ProfilePhase::TransformerOutputBackward:
        return "  Transformer output backward";

    case ProfilePhase::TransformerFinalNormBackward:
        return "  Transformer final norm backward";

    case ProfilePhase::TransformerBlocksBackward:
        return "  Transformer blocks backward";

    case ProfilePhase::BlockFFNBackward:
        return "    Block FFN backward";

    case ProfilePhase::FFNFlattenBackward:
        return "      FFN flatten backward";

    case ProfilePhase::FFNLinear2Backward:
        return "      FFN linear2 backward";

    case ProfilePhase::FFNGeluBackward:
        return "      FFN GELU backward";

    case ProfilePhase::FFNLinear1Backward:
        return "      FFN linear1 backward";

    case ProfilePhase::FFNUnflattenBackward:
        return "      FFN unflatten backward";

    case ProfilePhase::BlockNorm2Backward:
        return "    Block norm2 backward";

    case ProfilePhase::BlockResidual1Backward:
        return "    Block residual1 merge";

    case ProfilePhase::BlockAttentionBackward:
        return "    Block attention backward";

    case ProfilePhase::BlockNorm1Backward:
        return "    Block norm1 backward";

    case ProfilePhase::BlockInputMergeBackward:
        return "    Block input merge";

    case ProfilePhase::TransformerEmbeddingBackward:
        return "  Transformer embedding backward";

    case ProfilePhase::AttentionFlattenForward:
        return "      Attention flatten";

    case ProfilePhase::AttentionQProjectionForward:
        return "      Attention Q projection";

    case ProfilePhase::AttentionKProjectionForward:
        return "      Attention K projection";

    case ProfilePhase::AttentionVProjectionForward:
        return "      Attention V projection";

    case ProfilePhase::AttentionCacheForward:
        return "      Attention cache";

    case ProfilePhase::AttentionKernelForward:
        return "      Attention kernel";

    case ProfilePhase::AttentionOutputProjectionForward:
        return "      Attention output projection";

    case ProfilePhase::AttentionUnflattenForward:
        return "      Attention unflatten";

    case ProfilePhase::AttentionFlattenBackward:
        return "      Attention flatten backward";

    case ProfilePhase::AttentionOutputProjectionBackward:
        return "      Attention output proj back";

    case ProfilePhase::AttentionKernelBackward:
        return "      Attention kernel backward";

    case ProfilePhase::AttentionQProjectionBackward:
        return "      Attention Q projection backward";

    case ProfilePhase::AttentionKProjectionBackward:
        return "      Attention K projection backward";

    case ProfilePhase::AttentionVProjectionBackward:
        return "      Attention V projection backward";

    case ProfilePhase::AttentionGradientMergeBackward:
        return "      Attention gradient merge";

    case ProfilePhase::AttentionUnflattenBackward:
        return "      Attention unflatten backward";

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

    case ProfilePhase::Epoch:
        return "Epoch";

    case ProfilePhase::TotalTraining:
        return "Total training";

    default:
        return "Unknown";
    }
}