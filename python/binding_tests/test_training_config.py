import transformer_toy as tt


def assert_close(actual: float, expected: float) -> None:
    assert abs(actual - expected) < 1e-6


def main() -> None:
    # --------------------------------------------------------------
    # Default construction
    # --------------------------------------------------------------
    default_config = tt.TrainingConfig()

    print("Default configuration:")
    print(f"  epochs: {default_config.epochs}")
    print(f"  batch_size: {default_config.batch_size}")
    print(f"  device: {default_config.device}")
    print(f"  run_name: {default_config.run_name}")
    print(
        "  generation_config.max_new_tokens: "
        f"{default_config.generation_config.max_new_tokens}"
    )

    assert default_config.epochs == 1
    assert default_config.batch_size == 1
    assert default_config.device == tt.Device.AUTO

    assert default_config.log_every_steps == 0
    assert default_config.run_name == "default_run"

    assert default_config.enable_checkpointing is False
    assert default_config.enable_best_checkpoint is False
    assert default_config.checkpoint_every_epochs == 1
    assert_close(default_config.best_checkpoint_min_delta, 0.0)
    assert default_config.checkpoint_directory == "checkpoints"

    assert default_config.enable_early_stopping is False
    assert default_config.early_stopping_patience == 5
    assert_close(default_config.early_stopping_min_delta, 0.0)
    assert default_config.prefer_validation_loss is True

    assert default_config.enable_lr_scheduler is False
    assert default_config.lr_schedule == tt.LearningRateSchedule.STEP_DECAY
    assert default_config.lr_step_size == 1
    assert_close(default_config.lr_gamma, 1.0)
    assert_close(default_config.minimum_learning_rate, 0.0)

    assert default_config.enable_generation_snapshots is False
    assert default_config.generation_every_epochs == 1
    assert default_config.generation_prompt == ""

    assert default_config.enable_profiling is False
    assert default_config.print_profile_summary is True
    assert default_config.synchronize_profiling_phases is False
    assert default_config.profile_warmup_steps == 5
    assert default_config.profile_every_steps == 1

    default_config.validate()

    # --------------------------------------------------------------
    # Configure a realistic training run
    # --------------------------------------------------------------
    config = tt.TrainingConfig()

    config.epochs = 25
    config.batch_size = 32
    config.device = tt.Device.CUDA

    config.log_every_steps = 50
    config.run_name = "tiny_shakespeare_python"

    config.enable_checkpointing = True
    config.enable_best_checkpoint = True
    config.checkpoint_every_epochs = 2
    config.best_checkpoint_min_delta = 0.001
    config.checkpoint_directory = "python_checkpoints"

    config.enable_early_stopping = True
    config.early_stopping_patience = 4
    config.early_stopping_min_delta = 0.0005
    config.prefer_validation_loss = True

    config.enable_lr_scheduler = True
    config.lr_schedule = tt.LearningRateSchedule.COSINE_DECAY
    config.lr_step_size = 5
    config.lr_gamma = 0.9
    config.minimum_learning_rate = 1e-6

    generation = tt.GenerationConfig(
        max_new_tokens=200,
        temperature=0.8,
        top_k=40,
        greedy_sampling=False,
        print_generated_text=True,
        print_token_ids=False,
        random_seed=1234,
    )

    config.enable_generation_snapshots = True
    config.generation_every_epochs = 2
    config.generation_config = generation
    config.generation_prompt = "ROMEO:"

    config.enable_profiling = True
    config.print_profile_summary = True
    config.synchronize_profiling_phases = True
    config.profile_warmup_steps = 10
    config.profile_every_steps = 25

    print("\nConfigured training run:")
    print(f"  epochs: {config.epochs}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  device: {config.device}")
    print(f"  run_name: {config.run_name}")
    print(f"  lr_schedule: {config.lr_schedule}")
    print(f"  generation_prompt: {config.generation_prompt}")
    print(
        "  generation temperature: "
        f"{config.generation_config.temperature}"
    )

    assert config.epochs == 25
    assert config.batch_size == 32
    assert config.device == tt.Device.CUDA

    assert config.log_every_steps == 50
    assert config.run_name == "tiny_shakespeare_python"

    assert config.enable_checkpointing is True
    assert config.enable_best_checkpoint is True
    assert config.checkpoint_every_epochs == 2
    assert_close(config.best_checkpoint_min_delta, 0.001)
    assert config.checkpoint_directory == "python_checkpoints"

    assert config.enable_early_stopping is True
    assert config.early_stopping_patience == 4
    assert_close(config.early_stopping_min_delta, 0.0005)
    assert config.prefer_validation_loss is True

    assert config.enable_lr_scheduler is True
    assert config.lr_schedule == tt.LearningRateSchedule.COSINE_DECAY
    assert config.lr_step_size == 5
    assert_close(config.lr_gamma, 0.9)
    assert_close(config.minimum_learning_rate, 1e-6)

    assert config.enable_generation_snapshots is True
    assert config.generation_every_epochs == 2
    assert config.generation_prompt == "ROMEO:"
    assert config.generation_config.max_new_tokens == 200
    assert_close(config.generation_config.temperature, 0.8)
    assert config.generation_config.top_k == 40
    assert config.generation_config.random_seed == 1234

    assert config.enable_profiling is True
    assert config.print_profile_summary is True
    assert config.synchronize_profiling_phases is True
    assert config.profile_warmup_steps == 10
    assert config.profile_every_steps == 25

    config.validate()

    print("\nTrainingConfig tests passed.")


if __name__ == "__main__":
    main()