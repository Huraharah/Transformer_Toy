import transformer_toy as tt


def main() -> None:
    # Default constructor
    default_config = tt.TransformerBlockConfig()

    print("Default configuration:")
    print(f"  d_model: {default_config.d_model}")
    print(f"  d_ff: {default_config.d_ff}")
    print(f"  Pre-norm: {default_config.pre_norm}")
    print(f"  Use bias: {default_config.use_bias}")
    print(f"  Residual dropout: {default_config.residual_dropout}")
    print(f"  FFN dropout: {default_config.ffn_dropout}")
    print(f"  Attention type: {default_config.attention_type}")
    print(f"  Num heads: {default_config.num_heads}")

    assert default_config.d_model == 1
    assert default_config.d_ff == 1
    assert default_config.pre_norm is True
    assert default_config.use_bias is True
    assert abs(default_config.residual_dropout) < 1e-6
    assert abs(default_config.ffn_dropout) < 1e-6
    assert default_config.attention_type == tt.AttentionType.SINGLE_HEAD
    assert default_config.num_heads == 1

    default_config.validate()

    # Parameterized constructor
    config = tt.TransformerBlockConfig(
        d_model=256,
        d_ff=1024,
        residual_dropout=0.10,
        ffn_dropout=0.20,
        pre_norm=False,
        use_bias=False,
    )

    # These are not part of the current C++ constructor.
    config.attention_type = tt.AttentionType.MULTI_HEAD
    config.num_heads = 8

    print("\nParameterized configuration:")
    print(f"  d_model: {config.d_model}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  Pre-norm: {config.pre_norm}")
    print(f"  Use bias: {config.use_bias}")
    print(f"  Residual dropout: {config.residual_dropout}")
    print(f"  FFN dropout: {config.ffn_dropout}")
    print(f"  Attention type: {config.attention_type}")
    print(f"  Num heads: {config.num_heads}")

    assert config.d_model == 256
    assert config.d_ff == 1024
    assert config.pre_norm is False
    assert config.use_bias is False
    assert abs(config.residual_dropout - 0.10) < 1e-6
    assert abs(config.ffn_dropout - 0.20) < 1e-6
    assert config.attention_type == tt.AttentionType.MULTI_HEAD
    assert config.num_heads == 8

    config.validate()

    # Read/write mutation test
    config.d_model = 512
    config.d_ff = 2048
    config.pre_norm = True
    config.use_bias = True
    config.residual_dropout = 0.15
    config.ffn_dropout = 0.25
    config.attention_type = tt.AttentionType.SINGLE_HEAD
    config.num_heads = 1

    assert config.d_model == 512
    assert config.d_ff == 2048
    assert config.pre_norm is True
    assert config.use_bias is True
    assert abs(config.residual_dropout - 0.15) < 1e-6
    assert abs(config.ffn_dropout - 0.25) < 1e-6
    assert config.attention_type == tt.AttentionType.SINGLE_HEAD
    assert config.num_heads == 1

    config.validate()

    print("\nTransformerBlockConfig tests passed.")


if __name__ == "__main__":
    main()