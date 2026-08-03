import numpy as np

import transformer_toy as tt


def create_config(
    attention_type: tt.AttentionType,
    d_model: int,
    d_ff: int,
    num_heads: int,
) -> tt.TransformerBlockConfig:
    config = tt.TransformerBlockConfig(
        d_model=d_model,
        d_ff=d_ff,
        residual_dropout=0.0,
        ffn_dropout=0.0,
        pre_norm=True,
        use_bias=True,
    )

    config.attention_type = attention_type
    config.num_heads = num_heads
    config.validate()

    return config


def test_block(
    attention_type: tt.AttentionType,
    d_model: int,
    d_ff: int,
    num_heads: int,
    input_values: np.ndarray,
    grad_output_values: np.ndarray,
    seed: int,
) -> None:
    config = create_config(
        attention_type=attention_type,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
    )

    cpu_block = tt.TransformerBlock(
        config=config,
        rng=tt.Random(seed),
    )

    print(cpu_block)
    print(f"Attention type: {attention_type}")

    parameters = cpu_block.parameters()

    """
        Attention:
            Q/K/V/output projections = 8 Parameters

        LayerNorm 1:
            gamma/beta = 2 Parameters

        FFN:
            two Linear layers = 4 Parameters

        LayerNorm 2:
            gamma/beta = 2 Parameters

        Total = 16 Parameters
    """

    assert len(parameters) == 16

    for parameter in parameters:
        assert isinstance(parameter, tt.Parameter)
        parameter.validate()

    # ----------------------------------------------------------
    # CPU forward
    # ----------------------------------------------------------
    cpu_input = tt.Tensor(input_values)
    cpu_output = cpu_block.forward(cpu_input)

    assert cpu_output.shape == list(input_values.shape)
    assert cpu_output.device == tt.Device.CPU

    assert np.all(np.isfinite(cpu_output.numpy()))

    print("\nCPU output:")
    print(cpu_output.numpy())

    # ----------------------------------------------------------
    # CPU backward
    # ----------------------------------------------------------
    cpu_grad_output = tt.Tensor(grad_output_values)

    cpu_grad_input = cpu_block.backward(
        cpu_grad_output
    )

    assert cpu_grad_input.shape == list(input_values.shape)
    assert cpu_grad_input.device == tt.Device.CPU

    assert np.all(np.isfinite(cpu_grad_input.numpy()))

    for parameter in parameters:
        gradient = parameter.grad.numpy()

        assert gradient.shape == tuple(
            parameter.value.shape
        )
        assert np.all(np.isfinite(gradient))

    print("\nCPU input gradient:")
    print(cpu_grad_input.numpy())

    # ----------------------------------------------------------
    # CUDA parity
    # ----------------------------------------------------------
    if tt.cuda_available():
        cuda_config = create_config(
            attention_type=attention_type,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
        )

        cuda_block = tt.TransformerBlock(
            config=cuda_config,
            rng=tt.Random(seed),
        )

        cuda_input = tt.Tensor(input_values)
        cuda_input.to_cuda()

        cuda_output = cuda_block.forward(cuda_input)

        assert cuda_output.device == tt.Device.CUDA

        np.testing.assert_allclose(
            cuda_output.numpy(),
            cpu_output.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

        cuda_grad_output = tt.Tensor(
            grad_output_values
        )
        cuda_grad_output.to_cuda()

        cuda_grad_input = cuda_block.backward(
            cuda_grad_output
        )

        assert cuda_grad_input.device == tt.Device.CUDA

        np.testing.assert_allclose(
            cuda_grad_input.numpy(),
            cpu_grad_input.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

        cuda_parameters = cuda_block.parameters()

        assert len(cuda_parameters) == len(parameters)

        for cpu_parameter, cuda_parameter in zip(
            parameters,
            cuda_parameters,
        ):
            np.testing.assert_allclose(
                cuda_parameter.grad.numpy(),
                cpu_parameter.grad.numpy(),
                rtol=1e-3,
                atol=1e-3,
            )

        print("\nCUDA parity passed.")


def test_failures(
    d_model: int,
    d_ff: int,
) -> None:
    config = create_config(
        attention_type=tt.AttentionType.MULTI_HEAD,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=2,
    )

    unused_block = tt.TransformerBlock(
        config=config,
        rng=tt.Random(99),
    )

    # Backward before forward
    try:
        unused_block.backward(
            tt.Tensor([1, 2, d_model], 1.0)
        )
        raise AssertionError(
            "Expected backward-before-forward to fail."
        )
    except RuntimeError:
        pass

    # Incorrect input rank
    try:
        unused_block.forward(
            tt.Tensor([2, d_model])
        )
        raise AssertionError(
            "Expected rank-2 input to fail."
        )
    except ValueError:
        pass

    # Incorrect embedding dimension
    try:
        unused_block.forward(
            tt.Tensor([1, 2, d_model + 1])
        )
        raise AssertionError(
            "Expected embedding-dimension mismatch to fail."
        )
    except ValueError:
        pass


def main() -> None:
    seed = 1234
    d_model = 4
    d_ff = 8

    input_values = np.array(
        [
            [
                [1.0, 0.0, 0.5, -1.0],
                [0.0, 1.0, -0.5, 0.5],
                [1.5, -1.0, 0.25, 0.75],
            ],
            [
                [-0.5, 1.0, 0.0, 1.5],
                [0.75, 0.25, -1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
        ],
        dtype=np.float32,
    )

    grad_output_values = np.arange(
        input_values.size,
        dtype=np.float32,
    ).reshape(input_values.shape) / 20.0

    print("Testing single-head TransformerBlock")

    test_block(
        attention_type=tt.AttentionType.SINGLE_HEAD,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=1,
        input_values=input_values,
        grad_output_values=grad_output_values,
        seed=seed,
    )

    print("\nTesting multi-head TransformerBlock")

    test_block(
        attention_type=tt.AttentionType.MULTI_HEAD,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=2,
        input_values=input_values,
        grad_output_values=grad_output_values,
        seed=seed,
    )

    test_failures(
        d_model=d_model,
        d_ff=d_ff,
    )

    print("\nTransformerBlock tests passed.")


if __name__ == "__main__":
    main()