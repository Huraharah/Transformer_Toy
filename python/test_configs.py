import transformer_toy as tt


def main() -> None:
    block = tt.TransformerBlockConfig(
        d_model=128,
        d_ff=512,
    )
    block.attention_type = tt.AttentionType.MULTI_HEAD
    block.num_heads = 4
    block.validate()

    model = tt.TransformerModelConfig(
        vocab_size=65,
        max_seq_len=128,
        num_layers=4,
        block=block,
    )
    model.validate()

    generation = tt.GenerationConfig(
        max_new_tokens=100,
        temperature=0.8,
        top_k=20,
    )
    generation.validate()

    training = tt.TrainingConfig()
    training.epochs = 10
    training.batch_size = 16
    training.device = tt.Device.CUDA
    training.generation_config = generation
    training.validate()

    print(f"CUDA available: {tt.cuda_available()}")
    print(f"Model layers: {model.num_layers}")
    print(f"Attention heads: {model.block.num_heads}")
    print(f"Training device: {training.device}")
    print(f"Generation tokens: {training.generation_config.max_new_tokens}")


if __name__ == "__main__":
    main()