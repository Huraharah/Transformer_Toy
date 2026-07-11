#pragma once
#include "training/trainer.h"
#include "training/training_config.h"
#include "training/checkpoint.h"
#include "training/epoch_callback.h"
#include "training/best_checkpoint_callback.h"
#include "training/checkpoint_callback.h"
#include "training/early_stopping_callback.h"
#include "training/learning_rate_scheduler_callback.h"
#include "training/generation_callback.h"


struct TrainingSession {
private:
	Trainer trainer;

	std::unique_ptr<CheckpointCallback> checkpointCallback;
	std::unique_ptr<BestCheckpointCallback> bestCheckpointCallback;
	std::unique_ptr<EarlyStoppingCallback> earlyStoppingCallback;
	std::unique_ptr<LearningRateSchedulerCallback> lrSchedulerCallback;
	std::unique_ptr<GenerationCallback> generationCallback;

public:
	TrainingSession(
		TrainableModel& model,
		Loss& loss,
		Optimizer& optimizer,
		const TrainingConfig& config,
		const CharTokenizer* tokenizer = nullptr
	);

	void train(
		const std::vector<TrainingBatch>& trainBatches,
		const std::vector<TrainingBatch>* validationBatches = nullptr
	);
	TrainingHistory& getHistory();
	const TrainingHistory& getHistory() const;
	GenerationCallback* getGenerationCallback();
	const GenerationCallback* getGenerationCallback() const;
};