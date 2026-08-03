from pathlib import Path
import tempfile

import numpy as np

import transformer_toy as tt


def main() -> None:
    corpus = (
        "To be, or not to be, that is the question.\n"
        "Whether tis nobler in the mind to suffer.\n"
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        text_path = root / "sample.txt"

        text_path.write_text(
            corpus,
            encoding="utf-8",
        )

        # ------------------------------------------------------
        # Default tokenizer constructor
        # ------------------------------------------------------
        default_dataset = tt.TextDataset(
            str(text_path),
            8,
        )

        print(default_dataset)
        print(f"Tokenizer: {default_dataset.tokenizer.name()}")
        print(f"Token count: {default_dataset.token_count}")
        print(f"Vocabulary size: {default_dataset.vocab_size}")
        print(f"Windows: {default_dataset.num_windows}")

        written_corpus = text_path.read_bytes().decode("utf-8")

        print(f"Written Corpus Length: {len(written_corpus)}")

        assert default_dataset.raw_text == written_corpus
        assert default_dataset.token_count == len(written_corpus)
        assert len(default_dataset.token_ids) == len(written_corpus)
        assert default_dataset.num_windows == (
            default_dataset.token_count - 8
        )
        assert len(default_dataset) == default_dataset.num_windows

        assert isinstance(
            default_dataset.tokenizer,
            tt.Tokenizer,
        )
        assert isinstance(
            default_dataset.tokenizer,
            tt.CharTokenizer,
        )
        assert (
            default_dataset.tokenizer.type()
            == tt.TokenizerType.CHARACTER
        )

        assert default_dataset.raw_text == written_corpus
        assert len(default_dataset.token_ids) == len(written_corpus)

        # ------------------------------------------------------
        # Sequential input and target windows
        # ------------------------------------------------------
        inputs = default_dataset.get_input_window(0)
        targets = default_dataset.get_target_window(0)

        assert inputs.shape == [1, 8]
        assert targets.shape == [1, 8]

        input_ids = inputs.numpy().astype(np.int64).reshape(-1)
        target_ids = targets.numpy().astype(np.int64).reshape(-1)

        assert np.array_equal(
            input_ids[1:],
            target_ids[:-1],
        )

        decoded_input = default_dataset.tokenizer.decode(
            input_ids.tolist()
        )
        decoded_target = default_dataset.tokenizer.decode(
            target_ids.tolist()
        )

        assert decoded_input == written_corpus[:8]
        assert decoded_target == written_corpus[1:9]

        # ------------------------------------------------------
        # Random batch
        # ------------------------------------------------------
        rng_a = tt.Random(1234)
        rng_b = tt.Random(1234)

        batch_inputs_a, batch_targets_a = (
            default_dataset.get_batch(4, rng_a)
        )

        batch_inputs_b, batch_targets_b = (
            default_dataset.get_batch(4, rng_b)
        )

        assert batch_inputs_a.shape == [4, 8]
        assert batch_targets_a.shape == [4, 8]

        np.testing.assert_array_equal(
            batch_inputs_a.numpy(),
            batch_inputs_b.numpy(),
        )
        np.testing.assert_array_equal(
            batch_targets_a.numpy(),
            batch_targets_b.numpy(),
        )

        batch_inputs_np = batch_inputs_a.numpy()
        batch_targets_np = batch_targets_a.numpy()

        np.testing.assert_array_equal(
            batch_inputs_np[:, 1:],
            batch_targets_np[:, :-1],
        )

        # ------------------------------------------------------
        # Configured tokenizer
        # ------------------------------------------------------
        tokenizer_config = tt.TokenizerConfig()
        tokenizer_config.type = tt.TokenizerType.BPE
        tokenizer_config.vocab_size = 64
        tokenizer_config.min_frequency = 1
        tokenizer_config.train_if_missing = True
        tokenizer_config.model_path = str(
            root / "tokenizers" / "sample_bpe.bin"
        )

        bpe_dataset = tt.TextDataset(
            str(text_path),
            4,
            tokenizer_config,
        )

        assert isinstance(
            bpe_dataset.tokenizer,
            tt.BPETokenizer,
        )
        assert bpe_dataset.tokenizer.is_trained is True
        assert bpe_dataset.vocab_size > 0
        assert Path(tokenizer_config.model_path).exists()

        # Reconstructing should load the saved tokenizer.
        loaded_bpe_dataset = tt.TextDataset(
            str(text_path),
            4,
            tokenizer_config,
        )

        assert (
            loaded_bpe_dataset.token_ids
            == bpe_dataset.token_ids
        )
        assert (
            loaded_bpe_dataset.vocab_size
            == bpe_dataset.vocab_size
        )

        # ------------------------------------------------------
        # Failure cases
        # ------------------------------------------------------
        try:
            tt.TextDataset(
                str(text_path),
                0,
            )
            raise AssertionError(
                "Expected zero context length to fail."
            )
        except ValueError:
            pass

        try:
            default_dataset.get_batch(
                0,
                tt.Random(1),
            )
            raise AssertionError(
                "Expected zero batch size to fail."
            )
        except ValueError:
            pass

        try:
            default_dataset.get_input_window(
                default_dataset.num_windows
            )
            raise AssertionError(
                "Expected invalid window index to fail."
            )
        except IndexError:
            pass

    print("TextDataset tests passed.")


if __name__ == "__main__":
    main()
