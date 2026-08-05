from pathlib import Path
import tempfile

import transformer_toy as tt


def make_model_and_tokenizer():
    corpus = "To be or not to be. To be again."

    tokenizer = tt.CharTokenizer()
    tokenizer.train(corpus)

    block = tt.TransformerBlockConfig(
        d_model=4,
        d_ff=8,
    )

    block.attention_type = (
        tt.AttentionType.MULTI_HEAD
    )
    block.num_heads = 2
    block.validate()

    model_config = tt.TransformerModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=8,
        num_layers=1,
        block=block,
    )

    model = tt.Transformer(
        config=model_config,
        rng=tt.Random(1234),
    )

    return model, tokenizer


def make_context(
    epoch: int,
    train_loss: float,
    validation_loss: float | None = None,
) -> tt.EpochContext:
    context = tt.EpochContext()

    context.epoch = epoch
    context.total_epochs = 10
    context.global_step = epoch * 100
    context.train_loss = train_loss
    context.device = tt.Device.CPU
    context.run_name = "callback_test"

    if validation_loss is not None:
        context.validation_loss = validation_loss
        context.has_validation_loss = True

    return context


def test_lr_scheduler(
    model,
    optimizer,
    history,
) -> None:
    callback = tt.LearningRateSchedulerCallback(
        schedule=tt.LearningRateSchedule.STEP_DECAY,
        step_size=2,
        gamma=0.5,
        minimum_learning_rate=0.001,
    )

    assert isinstance(callback, tt.EpochCallback)
    assert callback.initial_learning_rate == 0.0
    assert callback.current_learning_rate == 0.0

    callback.on_epoch_end(
        make_context(1, 3.0),
        model,
        optimizer,
        history,
    )

    assert abs(
        callback.initial_learning_rate - 0.1
    ) < 1e-6

    assert abs(
        optimizer.learning_rate - 0.1
    ) < 1e-6

    callback.on_epoch_end(
        make_context(2, 2.5),
        model,
        optimizer,
        history,
    )

    assert abs(
        optimizer.learning_rate - 0.05
    ) < 1e-6


def test_early_stopping(
    model,
    optimizer,
    history,
) -> None:
    callback = tt.EarlyStoppingCallback(
        patience=2,
        min_delta=0.01,
        prefer_validation_loss=True,
    )

    callback.on_epoch_end(
        make_context(1, 3.0, 3.2),
        model,
        optimizer,
        history,
    )

    assert callback.has_best_metric is True
    assert callback.best_epoch == 1
    assert callback.should_stop_training is False

    callback.on_epoch_end(
        make_context(2, 2.9, 3.2),
        model,
        optimizer,
        history,
    )

    assert callback.epochs_without_improvement == 1
    assert callback.should_stop_training is False

    callback.on_epoch_end(
        make_context(3, 2.8, 3.21),
        model,
        optimizer,
        history,
    )

    assert callback.epochs_without_improvement == 2
    assert callback.should_stop_training is True


def test_checkpoints(
    model,
    optimizer,
    history,
    root: Path,
) -> None:
    periodic_directory = root / "periodic"
    best_directory = root / "best"

    periodic_directory.mkdir()
    best_directory.mkdir()

    periodic = tt.CheckpointCallback(
        checkpoint_directory=str(periodic_directory),
        every_n_epochs=2,
    )

    periodic.on_epoch_end(
        make_context(1, 3.0),
        model,
        optimizer,
        history,
    )

    assert not (
        periodic_directory /
        "callback_test_epoch_1.bin"
    ).exists()

    periodic.on_epoch_end(
        make_context(2, 2.5),
        model,
        optimizer,
        history,
    )

    assert (
        periodic_directory /
        "callback_test_epoch_2.bin"
    ).exists()

    best = tt.BestCheckpointCallback(
        checkpoint_directory=str(best_directory),
        min_delta=0.01,
        prefer_validation_loss=True,
    )

    best.on_epoch_end(
        make_context(1, 3.0, 3.2),
        model,
        optimizer,
        history,
    )

    assert best.has_best_checkpoint is True
    assert best.best_epoch == 1
    assert Path(best.best_checkpoint_path).exists()

    best.on_epoch_end(
        make_context(2, 2.9, 3.25),
        model,
        optimizer,
        history,
    )

    assert best.best_epoch == 1

    best.on_epoch_end(
        make_context(3, 2.8, 3.0),
        model,
        optimizer,
        history,
    )

    assert best.best_epoch == 3
    assert abs(best.best_metric - 3.0) < 1e-6


def test_generation(
    model,
    tokenizer,
    optimizer,
    history,
) -> None:
    config = tt.GenerationConfig()
    config.max_new_tokens = 4
    config.temperature = 0.8
    config.top_k = min(
        5,
        tokenizer.vocab_size,
    )
    config.validate()

    callback = tt.GenerationCallback(
        tokenizer=tokenizer,
        prompt="To",
        generation_config=config,
        every_n_epochs=2,
        print_generated_text=False,
        random_seed=1234,
    )

    callback.on_epoch_end(
        make_context(1, 3.0),
        model,
        optimizer,
        history,
    )

    assert callback.snapshots == []

    callback.on_epoch_end(
        make_context(2, 2.5),
        model,
        optimizer,
        history,
    )

    snapshots = callback.snapshots

    assert len(snapshots) == 1
    assert snapshots[0].epoch == 2
    assert isinstance(snapshots[0].text, str)
    assert len(snapshots[0].text) >= 2

    assert callback.latest_snapshot.epoch == 2

    callback.clear_snapshots()

    assert callback.snapshots == []

    try:
        _ = callback.latest_snapshot

        raise AssertionError(
            "Expected missing snapshot to fail."
        )
    except RuntimeError:
        pass


def test_constructor_failures(
    tokenizer,
) -> None:
    try:
        tt.EarlyStoppingCallback(
            patience=-1,
        )
        raise AssertionError(
            "Expected negative patience to fail."
        )
    except ValueError:
        pass

    try:
        tt.BestCheckpointCallback(
            checkpoint_directory=".",
            min_delta=-0.1,
        )
        raise AssertionError(
            "Expected negative min_delta to fail."
        )
    except ValueError:
        pass

    try:
        tt.LearningRateSchedulerCallback(
            schedule=tt.LearningRateSchedule.STEP_DECAY,
            step_size=0,
        )
        raise AssertionError(
            "Expected zero step size to fail."
        )
    except ValueError:
        pass

    try:
        tt.GenerationCallback(
            tokenizer=tokenizer,
            prompt="",
            generation_config=tt.GenerationConfig(),
        )
        raise AssertionError(
            "Expected empty prompt to fail."
        )
    except ValueError:
        pass


def main() -> None:
    model, tokenizer = make_model_and_tokenizer()

    optimizer = tt.SGDOptimizer(
        learning_rate=0.1,
    )

    history = tt.TrainingHistory()
    history.add_train_loss(3.0)

    test_lr_scheduler(
        model,
        optimizer,
        history,
    )

    # Reset after scheduler mutation.
    optimizer.learning_rate = 0.1

    test_early_stopping(
        model,
        optimizer,
        history,
    )

    with tempfile.TemporaryDirectory() as directory:
        test_checkpoints(
            model,
            optimizer,
            history,
            Path(directory),
        )

    test_generation(
        model,
        tokenizer,
        optimizer,
        history,
    )

    test_constructor_failures(tokenizer)

    print("\nCallback tests passed.")


if __name__ == "__main__":
    main()