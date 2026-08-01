import transformer_toy as tt


def main() -> None:
    # Test the true default constructor.
    default_config = tt.TokenizerConfig()

    print("Default configuration:")
    print(f"  Type: {default_config.type}")
    print(f"  Vocabulary size: {default_config.vocab_size}")
    print(f"  Minimum frequency: {default_config.min_frequency}")
    print(f"  Model path: {default_config.model_path!r}")
    print(f"  Train if missing: {default_config.train_if_missing}")
    print(f"  Preserve whitespace: {default_config.preserve_whitespace}")
    print(f"  Preserve punctuation: {default_config.preserve_punctuation}")

    assert default_config.type == tt.TokenizerType.CHARACTER
    assert default_config.vocab_size == 512
    assert default_config.min_frequency == 2
    assert default_config.model_path == ""
    assert default_config.train_if_missing is True
    assert default_config.preserve_whitespace is True
    assert default_config.preserve_punctuation is True

    # Test the parameterized constructor.
    bpe_config = tt.TokenizerConfig(
        type=tt.TokenizerType.BPE,
        vocab_size=4096,
        min_frequency=3,
        model_path="tokenizers/shakespeare.bpe",
        train_if_missing=False,
        preserve_whitespace=False,
        preserve_punctuation=False,
    )

    print("\nParameterized BPE configuration:")
    print(f"  Type: {bpe_config.type}")
    print(f"  Vocabulary size: {bpe_config.vocab_size}")
    print(f"  Minimum frequency: {bpe_config.min_frequency}")
    print(f"  Model path: {bpe_config.model_path!r}")
    print(f"  Train if missing: {bpe_config.train_if_missing}")
    print(f"  Preserve whitespace: {bpe_config.preserve_whitespace}")
    print(f"  Preserve punctuation: {bpe_config.preserve_punctuation}")

    assert bpe_config.type == tt.TokenizerType.BPE
    assert bpe_config.vocab_size == 4096
    assert bpe_config.min_frequency == 3
    assert bpe_config.model_path == "tokenizers/shakespeare.bpe"
    assert bpe_config.train_if_missing is False
    assert bpe_config.preserve_whitespace is False
    assert bpe_config.preserve_punctuation is False

    # Test Python-side field mutation.
    bpe_config.type = tt.TokenizerType.WORD
    bpe_config.vocab_size = 2048
    bpe_config.min_frequency = 5
    bpe_config.model_path = "tokenizers/shakespeare.word"
    bpe_config.train_if_missing = True
    bpe_config.preserve_whitespace = True
    bpe_config.preserve_punctuation = True

    assert bpe_config.type == tt.TokenizerType.WORD
    assert bpe_config.vocab_size == 2048
    assert bpe_config.min_frequency == 5
    assert bpe_config.model_path == "tokenizers/shakespeare.word"
    assert bpe_config.train_if_missing is True
    assert bpe_config.preserve_whitespace is True
    assert bpe_config.preserve_punctuation is True

    print("\nTokenizerConfig tests passed.")


if __name__ == "__main__":
    main()