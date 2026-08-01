import transformer_toy as tt


def main() -> None:
    # ------------------------------------------------------------------
    # Default constructor
    # ------------------------------------------------------------------
    default_config = tt.AttentionConfig()

    print("Default configuration:")
    print(f"  Embed dim: {default_config.embed_dim}")
    print(f"  Num heads: {default_config.num_heads}")
    print(f"  Head dim: {default_config.head_dim}")
    print(f"  Causal: {default_config.causal}")
    print(f"  Use bias: {default_config.use_bias}")
    print(f"  Attention dropout: {default_config.attention_dropout}")
    print(f"  Projection dropout: {default_config.projection_dropout}")

    # ------------------------------------------------------------------
    # Parameterized constructor
    # ------------------------------------------------------------------
    config = tt.AttentionConfig(
        embed_dim=512,
        num_heads=8,
        causal=True,
        use_bias=False,
        attention_dropout=0.10,
        projection_dropout=0.20,
    )

    print("\nParameterized configuration:")
    print(f"  Embed dim: {config.embed_dim}")
    print(f"  Num heads: {config.num_heads}")
    print(f"  Head dim: {config.head_dim}")
    print(f"  Causal: {config.causal}")
    print(f"  Use bias: {config.use_bias}")
    print(f"  Attention dropout: {config.attention_dropout}")
    print(f"  Projection dropout: {config.projection_dropout}")

    assert config.embed_dim == 512
    assert config.num_heads == 8
    assert config.head_dim == 64
    assert config.causal is True
    assert config.use_bias is False
    assert abs(config.attention_dropout - 0.10) < 1e-6
    assert abs(config.projection_dropout - 0.20) < 1e-6

    # ------------------------------------------------------------------
    # Test read/write members
    # ------------------------------------------------------------------
    config.causal = False
    config.use_bias = True
    config.attention_dropout = 0.30
    config.projection_dropout = 0.40

    assert config.causal is False
    assert config.use_bias is True
    assert abs(config.attention_dropout - 0.30) < 1e-6
    assert abs(config.projection_dropout - 0.40) < 1e-6

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    config.validate()

    print("\nAttentionConfig tests passed.")


if __name__ == "__main__":
    main()