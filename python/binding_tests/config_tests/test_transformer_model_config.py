import transformer_toy as tt

def main() -> None:
    # Default constructor

    default_config = tt.TransformerModelConfig()

    print("Default configuration:")
    print(f"  vocab_size: {default_config.vocab_size}")
    print(f"  max_seq_len: {default_config.max_seq_len}")
    print(f"  num_layers: {default_config.num_layers}")
    print(f"  block.d_model: {default_config.block.d_model}")
    print(f"  block.d_ff: {default_config.block.d_ff}")
    print(f"  learned_positional_embeddings: {default_config.learned_positional_embeddings}")
    print(f"  embedding_dropout: {default_config.embedding_dropout}")

    assert default_config.vocab_size == 1
    assert default_config.max_seq_len == 1
    assert default_config.num_layers == 1
    assert default_config.block.d_model == 1
    assert default_config.block.d_ff == 1
    assert default_config.learned_positional_embeddings is True
    assert default_config.embedding_dropout < 1e-6

    default_config.validate()

    # Parameterized constructor

    block_config = tt.TransformerBlockConfig(
        d_model=256,
        d_ff=1024,
        residual_dropout=0.1,
        ffn_dropout=0.2,
        pre_norm=False,
        use_bias=False,
    )

    block_config.attention_type = tt.AttentionType.MULTI_HEAD
    block_config.num_heads = 8
    block_config.validate()

    parameterized_config = tt.TransformerModelConfig(
        vocab_size=256,
        max_seq_len=64,
        num_layers=2,
        block=block_config,
        learned_positional_embeddings=True,
        embedding_dropout=0.1,
    )

    print("Parameterized configuration:")
    print(f"  vocab_size: {parameterized_config.vocab_size}")
    print(f"  max_seq_len: {parameterized_config.max_seq_len}")
    print(f"  num_layers: {parameterized_config.num_layers}")
    print(f"  block.d_model: {parameterized_config.block.d_model}")
    print(f"  block.d_ff: {parameterized_config.block.d_ff}")
    print(
        "  block.attention_type: "
        f"{parameterized_config.block.attention_type}"
    )
    print(f"  block.num_heads: {parameterized_config.block.num_heads}")
    print(f"  learned_positional_embeddings: {parameterized_config.learned_positional_embeddings}")
    print(f"  embedding_dropout: {parameterized_config.embedding_dropout}")

    assert parameterized_config.vocab_size == 256
    assert parameterized_config.max_seq_len == 64
    assert parameterized_config.num_layers == 2
    assert parameterized_config.block.d_model == 256
    assert parameterized_config.block.d_ff == 1024
    assert (
        parameterized_config.block.attention_type
        == tt.AttentionType.MULTI_HEAD
    )
    assert parameterized_config.block.num_heads == 8
    assert parameterized_config.learned_positional_embeddings is True
    assert abs(parameterized_config.embedding_dropout - 0.1) < 1e-6

    parameterized_config.validate()

    # Read/write mutation test
    parameterized_config.vocab_size = 1024
    parameterized_config.max_seq_len = 128
    parameterized_config.num_layers = 4
    parameterized_config.learned_positional_embeddings = False
    parameterized_config.embedding_dropout = 0.5

    replacement_block = tt.TransformerBlockConfig(
        d_model=512,
        d_ff=2048,
    )
    parameterized_config.block = replacement_block

    assert parameterized_config.vocab_size == 1024
    assert parameterized_config.max_seq_len == 128
    assert parameterized_config.num_layers == 4
    assert parameterized_config.block.d_model == 512
    assert parameterized_config.block.d_ff == 2048
    assert parameterized_config.learned_positional_embeddings == False
    assert abs(parameterized_config.embedding_dropout - 0.5) < 1e-6

    parameterized_config.validate()

    print("\nTransformerModelConfig tests passed.")

if __name__ == "__main__":
    main()