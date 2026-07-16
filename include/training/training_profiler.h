#pragma once

#include "core/tensor.h"

#include <chrono>
#include <cstddef>
#include <iosfwd>
#include <string>
#include <unordered_map>


/*
    Forward declarations.

    ScopedProfile only stores a TrainingProfiler pointer, so a forward
    declaration is sufficient at that point.

    ScopedModelProfiler only stores a TrainableModel reference. Its
    constructor/destructor implementations must be in the .cpp file,
    where TrainableModel is fully defined.
*/
class TrainingProfiler;
class TrainableModel;


enum class ProfilePhase {
    BatchTransfer,

    ModelForward,
    TransformerEmbedding,
    TransformerBlocksForward,
    TransformerFinalNorm,
    TransformerOutputProjection,

    LossForward,
    LossBackward,

    ModelBackward,
    TransformerOutputBackward,
    TransformerFinalNormBackward,
    TransformerBlocksBackward,
    TransformerEmbeddingBackward,

    OptimizerStep,
    ZeroGrad,

    BlockNorm1Forward,
    BlockAttentionForward,
    BlockResidual1Forward,
    BlockNorm2Forward,
    BlockFFNForward,
    BlockResidual2Forward,

    BlockFFNBackward,
    BlockNorm2Backward,
    BlockResidual1Backward,
    BlockAttentionBackward,
    BlockNorm1Backward,
    BlockInputMergeBackward,

    FFNFlattenForward,
    FFNLinear1Forward,
    FFNGeluForward,
    FFNLinear2Forward,
    FFNUnflattenForward,

    FFNFlattenBackward,
    FFNLinear2Backward,
    FFNGeluBackward,
    FFNLinear1Backward,
    FFNUnflattenBackward,

    AttentionFlattenForward,
    AttentionQProjectionForward,
    AttentionKProjectionForward,
    AttentionVProjectionForward,
    AttentionCacheForward,
    AttentionKernelForward,
    AttentionOutputProjectionForward,
    AttentionUnflattenForward,

    AttentionFlattenBackward,
    AttentionOutputProjectionBackward,
    AttentionKernelBackward,
    AttentionQProjectionBackward,
    AttentionKProjectionBackward,
    AttentionVProjectionBackward,
    AttentionGradientMergeBackward,
    AttentionUnflattenBackward,

    TrainingStep,
    Validation,
    Callbacks,
    Epoch,
    TotalTraining
};


struct ProfileStats {
    double totalMilliseconds = 0.0;
    double minimumMilliseconds = 0.0;
    double maximumMilliseconds = 0.0;

    std::size_t callCount = 0;

    void addSample(
        double milliseconds
    );

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

    /*
        Controls whether nested ScopedProfile objects synchronize CUDA
        before recording the end of a phase.
    */
    bool synchronizeCudaPhases_ = false;

public:
    explicit TrainingProfiler(
        bool enabled = true
    );

    void setEnabled(
        bool enabled
    );

    bool isEnabled() const;

    void setSynchronizeCudaPhases(
        bool enabled
    );

    bool synchronizeCudaPhases() const;

    void start(
        ProfilePhase phase
    );

    void stop(
        ProfilePhase phase
    );

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


class ScopedProfile {
private:
    TrainingProfiler* profiler_ = nullptr;
    ProfilePhase phase_;
    Device device_;
    bool active_ = false;

public:
    ScopedProfile(
        TrainingProfiler* profiler,
        ProfilePhase phase,
        Device device
    );

    ~ScopedProfile();

    ScopedProfile(
        const ScopedProfile&
    ) = delete;

    ScopedProfile& operator=(
        const ScopedProfile&
        ) = delete;
};


class ScopedModelProfiler {
private:
    TrainableModel& model_;

public:
    ScopedModelProfiler(
        TrainableModel& model,
        TrainingProfiler* profiler
    );

    ~ScopedModelProfiler();

    ScopedModelProfiler(
        const ScopedModelProfiler&
    ) = delete;

    ScopedModelProfiler& operator=(
        const ScopedModelProfiler&
        ) = delete;
};


std::string profilePhaseToString(
    ProfilePhase phase
);