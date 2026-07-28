from pathlib import Path
import tempfile

import transformer_toy as tt


def main() -> None:
    corpus = "hello world\n"
    sample = "hello"

    # ----------------------------------------------------------
    # Factory creation
    # ----------------------------------------------------------
    config = tt.TokenizerConfig()
    config.type = tt.TokenizerType.CHARACTER

    tokenizer = tt.create_tokenizer(config)

    print("Created tokenizer:")
    print(f"  Python type: {type(tokenizer).__name__}")
    print(f"  Name: {tokenizer.name()}")
    print(f"  Type: {tokenizer.type()}")
    print(f"  Is trained: {tokenizer.is_trained}")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    assert isinstance(tokenizer, tt.Tokenizer)
    assert isinstance(tokenizer, tt.CharTokenizer)
    assert tokenizer.name() == "character"
    assert tokenizer.type() == tt.TokenizerType.CHARACTER
    assert tokenizer.is_trained is False
    assert tokenizer.vocab_size == 0

    # ----------------------------------------------------------
    # Training and round-trip encoding
    # ----------------------------------------------------------
    tokenizer.train(corpus)

    assert tokenizer.is_trained is True
    assert tokenizer.vocab_size > 0

    token_ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(token_ids)

    print("\nEncoding:")
    print(f"  Input: {sample!r}")
    print(f"  Token IDs: {token_ids}")
    print(f"  Decoded: {decoded!r}")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    assert isinstance(token_ids, list)
    assert all(isinstance(token_id, int) for token_id in token_ids)
    assert decoded == sample

    # ----------------------------------------------------------
    # Character-specific methods
    # ----------------------------------------------------------
    h_id = tokenizer.char_to_id("h")
    h_character = tokenizer.id_to_char(h_id)

    assert h_character == "h"

    print("\nCharacter lookup:")
    print(f"  'h' token ID: {h_id}")
    print(f"  Token {h_id}: {h_character!r}")

    # ----------------------------------------------------------
    # Save and load
    # ----------------------------------------------------------
    with tempfile.TemporaryDirectory() as temporary_directory:
        tokenizer_path = (
            Path(temporary_directory) /
            "character_tokenizer.bin"
        )

        assert tokenizer.save(str(tokenizer_path)) is True
        assert tokenizer_path.exists()

        loaded = tt.CharTokenizer()

        assert loaded.is_trained is False
        assert loaded.load(str(tokenizer_path)) is True
        assert loaded.is_trained is True
        assert loaded.vocab_size == tokenizer.vocab_size
        assert loaded.encode(sample) == token_ids
        assert loaded.decode(token_ids) == sample

    print("\nTokenizer interface tests passed.")


if __name__ == "__main__":
    main()