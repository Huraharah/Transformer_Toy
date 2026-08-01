import transformer_toy as tt


def main() -> None:
    config = tt.GenerationConfig(
        max_new_tokens=100,
        temperature=0.8,
        top_k=20,
        greedy_sampling=False,
        print_generated_text=True,
        print_token_ids=False,
        random_seed=1234,
    )

    print(f"Max new tokens: {config.max_new_tokens}")
    print(f"Temperature: {config.temperature}")
    print(f"Top-k: {config.top_k}")
    print(f"Greedy sampling: {config.greedy_sampling}")
    print(f"Print text: {config.print_generated_text}")
    print(f"Print token IDs: {config.print_token_ids}")
    print(f"Random seed: {config.random_seed}")

    assert config.max_new_tokens == 100
    assert abs(config.temperature - 0.8) < 1e-6
    assert config.top_k == 20
    assert config.greedy_sampling is False
    assert config.random_seed == 1234

    config.temperature = 0.65
    config.top_k = 10

    assert abs(config.temperature - 0.65) < 1e-6
    assert config.top_k == 10

    config.validate()

    print("GenerationConfig tests passed.")


if __name__ == "__main__":
    main()