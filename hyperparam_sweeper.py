import transformer_toy as tt
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from dataclasses import dataclass
from itertools import product
import time
import csv
import json
import gc

if not tt.cuda_available():
    raise RuntimeError("CUDA is required for the full-model training run")

CORPUS_PATH = Path("./data/shakespeare_complete_cleaned.txt")

if not CORPUS_PATH.exists():
    raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}")

@dataclass
class SequentialBatchBuildResult:
    batches: list[tt.TrainingBatch]
    sequence_count: int
    input_token_count: int
    transition_count: int
    tail_sequence_length: int

def build_sequential_batches(
    token_ids: list[int],
    range_begin: int,
    range_end: int,
    context_length: int,
    batch_size: int,
) -> SequentialBatchBuildResult:
    if context_length <= 0:
        raise ValueError("context_length must be greater than zero")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if range_begin < 0:
        raise ValueError("range_begin cannot be negative")
    if range_begin >= range_end:
        raise ValueError("Sequential batch range must be non-empty")
    if range_end > len(token_ids):
        raise IndexError("Sequential batch range exceeds token count")

    range_token_count = range_end - range_begin

    if range_token_count < 2:
        raise ValueError("Sequential batch range must contain at least two tokens.")

    usable_transitions = range_token_count - 1
    full_sequence_count = usable_transitions // context_length
    tail_length = usable_transitions % context_length
    range_tokens = np.asarray(token_ids[range_begin:range_end], dtype = np.float32)
    full_transition_count = full_sequence_count * context_length

    batches: list[tt.TrainingBatch] = []

    if full_sequence_count > 0:
        all_inputs = range_tokens[:full_transition_count].reshape(full_sequence_count, context_length)

        all_targets = range_tokens[1:full_transition_count + 1].reshape(full_sequence_count, context_length)

        for sequence_start in range(0, full_sequence_count, batch_size):
            sequence_end = min(sequence_start + batch_size, full_sequence_count)

            batch_inputs = np.ascontiguousarray(all_inputs[sequence_start:sequence_end])
            batch_targets = np.ascontiguousarray(all_targets[sequence_start:sequence_end])

            batches.append(tt.TrainingBatch(inputs = tt.Tensor(batch_inputs), targets = tt.Tensor(batch_targets)))

    if tail_length > 0:
        tail_start = full_transition_count
        tail_inputs = np.ascontiguousarray(range_tokens[tail_start:tail_start+tail_length].reshape(1, tail_length))
        tail_targets = np.ascontiguousarray(range_tokens[tail_start + 1:tail_start + 1 + tail_length].reshape(1, tail_length))
        batches.append(tt.TrainingBatch(inputs = tt.Tensor(tail_inputs), targets = tt.Tensor(tail_targets)))

    sequence_count = full_sequence_count + (1 if tail_length > 0 else 0)
    transition_count = full_transition_count + tail_length

    if not batches:
        raise RuntimeError("Sequential batch builder produced no batches")
    if transition_count != usable_transitions:
        raise RuntimeError("Sequential batch builder did not cover all corpus transitions")

    return SequentialBatchBuildResult(batches = batches, sequence_count = sequence_count, input_token_count = transition_count, transition_count = transition_count, tail_sequence_length = tail_length)

SWEEP_DIRECTORY = Path("./sweeps")
SHARED_DIRECTORY = SWEEP_DIRECTORY / "shared"
TOKENIZER_PATH = SHARED_DIRECTORY / "shakespeare_bpe_512_shared.tok"

SWEEP_DIRECTORY.mkdir(parents = True, exist_ok = True)
SHARED_DIRECTORY.mkdir(parents = True, exist_ok = True)

CONTEXT_LENGTH = 64
BATCH_SIZE = 32
MAX_EPOCHS = 50

# Tokenizer configuration
token_config = tt.TokenizerConfig()
token_config.type = tt.TokenizerType.BPE
token_config.model_path = str(TOKENIZER_PATH)
token_config.vocab_size = 512
token_config.min_frequency = 2
token_config.train_if_missing = True
token_config.preserve_whitespace = True
token_config.preserve_punctuation = True

# Import data
dataset = tt.TextDataset(file_path = str(CORPUS_PATH), context_length = CONTEXT_LENGTH, tokenizer_config = token_config)
tokenizer = dataset.tokenizer
all_token_ids = tokenizer.encode(dataset.raw_text)

if len(all_token_ids) < 100:
    raise RuntimeError("Dataset unexpectedly small")

split_index = int(len(all_token_ids) * 0.90)
split_index = max(2, min(split_index, len(all_token_ids) - 2))

training_build = build_sequential_batches(token_ids = all_token_ids, range_begin = 0, range_end = split_index, context_length = CONTEXT_LENGTH, batch_size = BATCH_SIZE)
validation_build = build_sequential_batches(token_ids = all_token_ids, range_begin = split_index, range_end = len(all_token_ids), context_length = CONTEXT_LENGTH, batch_size = BATCH_SIZE)

train_batches = training_build.batches
validation_batches = validation_build.batches

sweep_results: list[dict[str, object]] = []

def run_sweep( 
    dmodel: int,
    dff: int,
    numlayers: int,
    numheads: int,
    RUN_NAME: str,
    FULL_MODEL_DIRECTORY: Path,
    CHECKPOINT_DIRECTORY: Path
) -> dict[str, object]:
    trainer = None
    generation_callback = None
    lr_scheduler = None
    early_stopping = None
    best_checkpoint = None
    periodic_checkpoint = None
    model = None
    optimizer = None
    loss_function = None
    try:
        # Generation configuration
        gen_config = tt.GenerationConfig()
        gen_config.max_new_tokens = 50
        gen_config.temperature = 0.8
        gen_config.top_k = 20
        gen_config.random_seed = 1337
        gen_config.greedy_sampling = False
        gen_config.print_generated_text = True
        gen_config.print_token_ids = False
        gen_config.validate()  # Validate the configuration

        # TransformerBlock configuration
        trans_block_config = tt.TransformerBlockConfig()
        trans_block_config.d_model = dmodel
        trans_block_config.d_ff = dff
        trans_block_config.attention_type = tt.AttentionType.MULTI_HEAD
        trans_block_config.num_heads = numheads
        trans_block_config.pre_norm = True
        trans_block_config.use_bias = True
        trans_block_config.residual_dropout = 0.0
        trans_block_config.ffn_dropout = 0.0
        trans_block_config.validate()

        # TransformerModel configuration
        trans_model_config = tt.TransformerModelConfig()
        trans_model_config.vocab_size = dataset.vocab_size
        trans_model_config.max_seq_len = CONTEXT_LENGTH
        trans_model_config.num_layers = numlayers
        trans_model_config.block = trans_block_config
        trans_model_config.learned_positional_embeddings = True
        trans_model_config.embedding_dropout = 0.0
        trans_model_config.validate()

        # Training configuration
        train_config = tt.TrainingConfig()
        train_config.epochs = MAX_EPOCHS
        train_config.batch_size = BATCH_SIZE
        train_config.device = tt.Device.CUDA # Use GPU if available
        train_config.log_every_steps = 100
    
        for existing_path in CHECKPOINT_DIRECTORY.glob(
            f"{RUN_NAME}_*"
        ):
            existing_path.unlink()

        train_config.run_name = RUN_NAME
        train_config.enable_checkpointing = True
        train_config.enable_best_checkpoint = True
        train_config.checkpoint_every_epochs = 5
        train_config.best_checkpoint_min_delta = 0.0001
        train_config.checkpoint_directory = str(CHECKPOINT_DIRECTORY)
        train_config.enable_early_stopping = True
        train_config.early_stopping_patience = 10
        train_config.early_stopping_min_delta = 0.0001
        train_config.prefer_validation_loss = True
        train_config.enable_lr_scheduler = True
        train_config.lr_schedule = tt.LearningRateSchedule.COSINE_DECAY
        train_config.lr_step_size = 1
        train_config.lr_gamma = 1.0
        train_config.minimum_learning_rate = 1e-6
        train_config.enable_generation_snapshots = True
        train_config.generation_every_epochs = 10
        train_config.generation_config = gen_config
        train_config.generation_prompt = "To be, or not to be"
        train_config.enable_profiling = False
        train_config.validate()  # Validate the configuration

        # Build training tools and trainer session
        loss_function = tt.CrossEntropyLoss()
        optimizer = tt.AdamOptimizer(
            learning_rate = 0.001,
            beta1 = 0.9,
            beta2 = 0.999,
            epsilon = 1.0e-8,
            weight_decay = 0.0)
        model = tt.Transformer(
            config = trans_model_config,
            rng = tt.Random(420))
        trainer = tt.Trainer(
            model = model,
            loss_function = loss_function,
            optimizer = optimizer,
            config = train_config)

        # Build and attach callbacks
        periodic_checkpoint = tt.CheckpointCallback(
            checkpoint_directory = str(CHECKPOINT_DIRECTORY),
            every_n_epochs = 5)
        best_checkpoint = tt.BestCheckpointCallback(
            checkpoint_directory = str(CHECKPOINT_DIRECTORY),
            min_delta = 0.0001,
            prefer_validation_loss = True)
        early_stopping = tt.EarlyStoppingCallback(
            patience = 10,
            min_delta = 0.0001,
            prefer_validation_loss = True)
        lr_scheduler = tt.LearningRateSchedulerCallback(
            schedule = tt.LearningRateSchedule.COSINE_DECAY,
            step_size = 1,
            gamma = 1.0,
            minimum_learning_rate = 1e-6)
        generation_callback = tt.GenerationCallback(
            tokenizer = tokenizer,
            prompt = "To be, or not to be",
            generation_config = gen_config,
            every_n_epochs = 10,
            print_generated_text = True,
            random_seed = 2513)
        trainer.add_callback(periodic_checkpoint)
        trainer.add_callback(best_checkpoint)
        trainer.add_callback(early_stopping)
        trainer.add_callback(lr_scheduler)
        trainer.add_callback(generation_callback)

        # Print out full run configuration

        print("\n" + "=" * 60)
        print(f"Hyperparam Sweep Run Configuration: {RUN_NAME}")
        print("=" * 60)
        print("CUDA available:", tt.cuda_available())
        print("Raw characters:", len(dataset.raw_text))
        print("Encoded tokens:", len(all_token_ids))
        print("Vocabulary size:", dataset.vocab_size)
        print("Training batches:", len(train_batches))
        print("Validation batches:", len(validation_batches))
        print("Training transitions:", training_build.transition_count)
        print("Validation transitions:", validation_build.transition_count)
        print("Training tail length:", training_build.tail_sequence_length)
        print("Validation tail length:", validation_build.tail_sequence_length)
        print(f"Context length: {CONTEXT_LENGTH}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"d_model: {dmodel}")
        print(f"d_ff: {dff}")
        print(f"ffn_ratio: {dff / dmodel}")
        print(f"Layers: {numlayers}")
        print(f"Heads: {numheads}")
        print(f"Maximum epochs: {MAX_EPOCHS}")
        print("=" * 60, flush=True)

        # Run the trainer under timer

        training_start = time.perf_counter()

        trainer.train(train_batches = train_batches, validation_batches = validation_batches)

        training_elapsed = time.perf_counter() - training_start
        history = trainer.history
        completed_epochs = history.validation_count
        print("\n" + "=" * 60)
        print("Full Model Statistics")
        print("=" * 60)
        print("Device: CUDA")
        print("Completed epochs:", completed_epochs)
        print("Training steps:", history.train_count)
        print(f"Elapsed seconds: {training_elapsed:.3f}")
        print(f"Seconds per epoch: {training_elapsed / max(1, completed_epochs):.3f}")
        print("Initial training loss:", history.train_losses[0])
        print("Final training loss:", history.train_losses[-1])
        print("Initial validation loss:", history.validation_losses[0])
        print("Final validation loss:", history.validation_losses[-1])
        print("Initial learning rate:", lr_scheduler.initial_learning_rate)
        print("Final learning rate:", lr_scheduler.current_learning_rate)
        print("Best checkpoint epoch:", best_checkpoint.best_epoch)
        print("Best checkpoint path:", best_checkpoint.best_checkpoint_path)
        print("=" * 60)

        # Plot losses

        train_losses = np.asarray(history.train_losses, dtype=np.float32)
        validation_losses = np.asarray(history.validation_losses, dtype=np.float32)
        steps_per_epoch = len(train_batches)

        epoch_train_losses = np.array(
            [
                train_losses[index:index + steps_per_epoch].mean()
                for index in range(0, len(train_losses), steps_per_epoch)
            ],
            dtype=np.float32,
        )

        plt.figure(figsize=(10, 6))
        plt.plot(
            np.arange(1, len(epoch_train_losses) + 1),
            epoch_train_losses,
            marker="o",
            label="Mean training loss",
        )
        plt.plot(
            np.arange(1, len(validation_losses) + 1),
            validation_losses,
            marker="o",
            label="Validation loss",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title(f"Transformer_Toy Sweep: {RUN_NAME} losses")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f"./sweeps/{RUN_NAME}/losses.png")
        plt.close()

        # Print generation snapshots

        if not generation_callback.snapshots:
            print("No generation snapshots were captured.")
        else:
            for snapshot in generation_callback.snapshots:
                print("\n" + "=" * 60)
                print(f"Generation snapshot — epoch {snapshot.epoch}")
                print("=" * 60)
                print(snapshot.text)

        ## Load the best checkpoint, and generate

        best_path = Path(best_checkpoint.best_checkpoint_path)

        if not best_path.exists():
            raise FileNotFoundError(f"Best checkpoint was not created: {best_path}")

        loaded_metadata = tt.CheckpointMetadata()
        loaded_history = tt.TrainingHistory()

        tt.Checkpoint.load(
            path=str(best_path),
            parameters=model.parameters(),
            metadata=loaded_metadata,
            history=loaded_history,
        )

        print("Loaded:", loaded_metadata)

        final_generation_config = tt.GenerationConfig()
        final_generation_config.max_new_tokens = 500
        final_generation_config.temperature = 0.8
        final_generation_config.top_k = 10
        final_generation_config.greedy_sampling = False
        final_generation_config.print_generated_text = True
        final_generation_config.print_token_ids = False
        final_generation_config.random_seed = 9876
        final_generation_config.validate()

        final_text = model.generate(
            prompt="To be, or not to be",
            tokenizer=tokenizer,
            config=final_generation_config,
            rng=tt.Random(9876),
        )

        print("\n" + "=" * 60)
        print("Best-checkpoint generation")
        print("=" * 60)
        print(final_text)
        print("=" * 60)

        generation_path = (
            FULL_MODEL_DIRECTORY
            / "best_generation.txt"
        )

        generation_path.write_text(
            final_text,
            encoding="utf-8",
        )

        # Save history for review

        history_path = Path(f"./sweeps/{RUN_NAME}/history.csv")
        history.save_csv(str(history_path))

        print("History:", history_path)
        print("Tokenizer:", TOKENIZER_PATH)
        print("Checkpoints:", CHECKPOINT_DIRECTORY)

        # Log best information to csv

        best_epoch = best_checkpoint.best_epoch

        best_epoch_start = (
            (best_epoch - 1) * len(train_batches)
        )
        best_epoch_end = (
            best_epoch * len(train_batches)
        )

        best_epoch_train_loss = float(
            np.mean(
                train_losses[
                    best_epoch_start:best_epoch_end
                ]
            )
        )

        result = {
            "run_name": RUN_NAME,
            "d_model": dmodel,
            "d_ff": dff,
            "ffn_ratio": dff / dmodel,
            "num_layers": numlayers,
            "num_heads": numheads,
            "context_length": CONTEXT_LENGTH,
            "batch_size": BATCH_SIZE,
            "completed_epochs": completed_epochs,
            "training_steps": history.train_count,
            "elapsed_seconds": training_elapsed,
            "seconds_per_epoch": (
                training_elapsed / max(1, completed_epochs)
            ),
            "initial_train_loss": history.train_losses[0],
            "final_train_loss": history.train_losses[-1],
            "best_epoch": best_checkpoint.best_epoch,
            "best_epoch_train_loss": best_epoch_train_loss,
            "generalization_gap": (
                best_checkpoint.best_metric
                - best_epoch_train_loss
            ),
            "best_validation_loss": best_checkpoint.best_metric,
            "final_validation_loss": history.validation_losses[-1],
            "final_learning_rate": lr_scheduler.current_learning_rate,
            "best_checkpoint_path": best_checkpoint.best_checkpoint_path,
        }

        snapshot_path = (
            FULL_MODEL_DIRECTORY
            / "generation_snapshots.txt"
        )

        with snapshot_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            for snapshot in generation_callback.snapshots:
                file.write(
                    f"{'=' * 60}\n"
                    f"Epoch {snapshot.epoch}\n"
                    f"{'=' * 60}\n"
                    f"{snapshot.text}\n\n"
        )

        return result

    finally:
        del trainer
        del generation_callback
        del lr_scheduler
        del early_stopping
        del best_checkpoint
        del periodic_checkpoint
        del model
        del optimizer
        del loss_function

        gc.collect()

def main():

    d_models = [64, 128]
    d_ffs = [128, 256, 512]
    num_layers_options = [2, 4]
    num_heads_options = [2, 4]

    for dmodel, dff, numlayers, numheads in product(
        d_models,
        d_ffs,
        num_layers_options,
        num_heads_options):

        RUN_NAME = (
            f"d{dmodel}_"
            f"ff{dff}_"
            f"l{numlayers}_"
            f"h{numheads}"
        )

        FULL_MODEL_DIRECTORY = SWEEP_DIRECTORY / f"{RUN_NAME}"
        CHECKPOINT_DIRECTORY = FULL_MODEL_DIRECTORY / "checkpoints"

        FULL_MODEL_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        CHECKPOINT_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )
        try:
            result = run_sweep(dmodel, dff, numlayers, numheads, RUN_NAME, FULL_MODEL_DIRECTORY, CHECKPOINT_DIRECTORY)
        except Exception as error:
            error_path = (FULL_MODEL_DIRECTORY / "error.txt")
            error_path.write_text(repr(error), encoding = "utf-8")
            print(f"[ RUN FAILURE ] {RUN_NAME}: {error}")
            continue

        with (
            FULL_MODEL_DIRECTORY / "summary.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
        sweep_results.append(result)

        summary_path = SWEEP_DIRECTORY / "sweep_summary.csv"

        with summary_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=sweep_results[0].keys(),
            )
            writer.writeheader()
            writer.writerows(sweep_results)

 

if __name__ == "__main__":
    main()