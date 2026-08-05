from pathlib import Path
import tempfile

import numpy as np

import transformer_toy as tt


def make_batches(
    dataset: tt.TextDataset,
    batch_size: int,
    count: int,
    seed: int,
) -> list[tt.TrainingBatch]:
    rng = tt.Random(seed)
    batches: list[tt.TrainingBatch] = []

    for _ in range(count):
        inputs, targets = dataset.get_batch(
            batch_size,
            rng,
        )

        batches.append(
            tt.TrainingBatch(
                inputs=inputs,
                targets=targets,
            )
        )

    return batches


def build_model(
    vocab_size: int,
    context_length: int,
    seed: int,
) -> tt.Transformer:
    block = tt.TransformerBlockConfig(
        d_model=8,
        d_ff=16,
        residual_dropout=0.0,
        ffn_dropout=0.0,
        pre_norm=True,
        use_bias=True,
    )

    block.attention_type = (
        tt.AttentionType.MULTI_HEAD
    )
    block.num_heads = 2
    block.validate()

    model_config = tt.TransformerModelConfig(
        vocab_size=vocab_size,
        max_seq_len=context_length,
        num_layers=1,
        block=block,
        learned_positional_embeddings=True,
        embedding_dropout=0.0,
    )

    model_config.validate()

    return tt.Transformer(
        config=model_config,
        rng=tt.Random(seed),
    )


def run_training_test(
    device: tt.Device,
) -> None:
    corpus = (
        "Once upon a midnight dreary, while I pondered, "
        "weak and weary.\n"
        "Over many a quaint and curious volume of forgotten lore.\n"
    ) * 8

    context_length = 8
    batch_size = 4
    train_batch_count = 6
    validation_batch_count = 2
    epochs = 3

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus_path = root / "poe_sample.txt"

        with corpus_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:
            file.write(corpus)

        tokenizer_config = tt.TokenizerConfig()
        tokenizer_config.type = (
            tt.TokenizerType.CHARACTER
        )

        dataset = tt.TextDataset(
            file_path=str(corpus_path),
            context_length=context_length,
            tokenizer_config=tokenizer_config,
        )

        train_batches = make_batches(
            dataset=dataset,
            batch_size=batch_size,
            count=train_batch_count,
            seed=100,
        )

        validation_batches = make_batches(
            dataset=dataset,
            batch_size=batch_size,
            count=validation_batch_count,
            seed=200,
        )

        model = build_model(
            vocab_size=dataset.vocab_size,
            context_length=context_length,
            seed=1234,
        )

        loss_function = tt.CrossEntropyLoss()

        optimizer = tt.AdamOptimizer(
            learning_rate=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1.0e-8,
            weight_decay=0.0,
        )

        config = tt.TrainingConfig()
        config.epochs = epochs
        config.batch_size = batch_size
        config.device = device
        config.log_every_steps = 0
        config.run_name = "python_trainer_test"

        config.enable_checkpointing = False
        config.enable_best_checkpoint = False
        config.enable_early_stopping = False
        config.enable_lr_scheduler = False
        config.enable_generation_snapshots = False
        config.enable_profiling = False

        config.validate()

        trainer = tt.Trainer(
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            config=config,
        )

        assert isinstance(model, tt.TrainableModel)
        assert isinstance(loss_function, tt.Loss)
        assert isinstance(optimizer, tt.Optimizer)

        parameters = model.parameters()

        before = [
            parameter.value.numpy().copy()
            for parameter in parameters
        ]

        trainer.train(
            train_batches=train_batches,
            validation_batches=validation_batches,
        )

        history = trainer.history

        print(
            f"\nDevice: {device}"
        )
        print(
            f"Training entries: "
            f"{history.train_count}"
        )
        print(
            f"Validation entries: "
            f"{history.validation_count}"
        )
        print(
            f"Initial train loss: "
            f"{history.train_losses[0]}"
        )
        print(
            f"Final train loss: "
            f"{history.train_losses[-1]}"
        )

        assert history.train_count == (
            epochs * train_batch_count
        )

        assert history.validation_count == epochs

        assert all(
            np.isfinite(loss)
            for loss in history.train_losses
        )

        assert all(
            np.isfinite(loss)
            for loss in history.validation_losses
        )

        assert history.latest_train_loss > 0.0
        assert history.latest_validation_loss > 0.0

        # At least one model parameter must have changed.
        after = [
            parameter.value.numpy()
            for parameter in parameters
        ]

        assert any(
            not np.allclose(
                original,
                updated,
            )
            for original, updated in zip(
                before,
                after,
            )
        )

        evaluation_loss = trainer.evaluate(
            validation_batches
        )

        assert np.isfinite(evaluation_loss)
        assert evaluation_loss > 0.0

        history_path = root / "history.csv"
        history.save_csv(str(history_path))

        assert history_path.exists()


def test_callbacks() -> None:
    corpus = "To be or not to be, that is the question.\n" * 8

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus_path = root / "callback_corpus.txt"

        with corpus_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:
            file.write(corpus)

        dataset = tt.TextDataset(
            str(corpus_path),
            6,
        )

        batches = make_batches(
            dataset=dataset,
            batch_size=2,
            count=3,
            seed=1,
        )

        model = build_model(
            vocab_size=dataset.vocab_size,
            context_length=6,
            seed=2,
        )

        loss_function = tt.CrossEntropyLoss()

        optimizer = tt.SGDOptimizer(
            learning_rate=0.05,
        )

        config = tt.TrainingConfig()
        config.epochs = 5
        config.batch_size = 2
        config.device = tt.Device.CPU
        config.run_name = "callback_trainer_test"

        trainer = tt.Trainer(
            model,
            loss_function,
            optimizer,
            config,
        )

        scheduler = (
            tt.LearningRateSchedulerCallback(
                schedule=(
                    tt.LearningRateSchedule.STEP_DECAY
                ),
                step_size=2,
                gamma=0.5,
                minimum_learning_rate=0.001,
            )
        )

        early_stopping = tt.EarlyStoppingCallback(
            patience=2,
            min_delta=1000.0,
            prefer_validation_loss=False,
        )

        trainer.add_callback(scheduler)
        trainer.add_callback(early_stopping)

        trainer.train(batches)

        # First epoch establishes the best metric; the following
        # two epochs fail the deliberately huge improvement test.
        assert trainer.history.train_count == 3 * 3
        assert early_stopping.should_stop_training is True
        assert early_stopping.epochs_without_improvement == 2

        assert scheduler.initial_learning_rate > 0.0
        assert scheduler.current_learning_rate > 0.0


def test_failures() -> None:
    model = build_model(
        vocab_size=10,
        context_length=4,
        seed=1,
    )

    trainer = tt.Trainer(
        model=model,
        loss_function=tt.CrossEntropyLoss(),
        optimizer=tt.SGDOptimizer(0.01),
        config=tt.TrainingConfig(),
    )

    try:
        trainer.train([])
        raise AssertionError(
            "Expected empty training batches to fail."
        )
    except ValueError:
        pass

    try:
        trainer.evaluate([])
        raise AssertionError(
            "Expected empty validation batches to fail."
        )
    except ValueError:
        pass

    try:
        trainer.train([object()])
        raise AssertionError(
            "Expected invalid batch type to fail."
        )
    except TypeError:
        pass


def main() -> None:
    run_training_test(
        tt.Device.CPU
    )

    if tt.cuda_available():
        run_training_test(
            tt.Device.CUDA
        )

    test_callbacks()
    test_failures()

    print("\nTrainer tests passed.")


if __name__ == "__main__":
    main()