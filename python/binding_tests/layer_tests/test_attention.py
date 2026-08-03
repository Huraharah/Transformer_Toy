import numpy as np

import transformer_toy as tt


def assert_parameter_layout(parameters, embed_dim: int) -> None:
    assert len(parameters) == 8

    expected_shapes = [
        [embed_dim, embed_dim],
        [embed_dim],
        [embed_dim, embed_dim],
        [embed_dim],
        [embed_dim, embed_dim],
        [embed_dim],
        [embed_dim, embed_dim],
        [embed_dim],
    ]

    expected_names = [
        "linear.weight",
        "linear.bias",
        "linear.weight",
        "linear.bias",
        "linear.weight",
        "linear.bias",
        "linear.weight",
        "linear.bias",
    ]

    for parameter, shape, name in zip(
        parameters,
        expected_shapes,
        expected_names,
    ):
        assert isinstance(parameter, tt.Parameter)
        assert parameter.value.shape == shape
        assert parameter.grad.shape == shape
        assert parameter.name == name


def test_self_attention(
    input_values: np.ndarray,
    grad_output_values: np.ndarray,
    embed_dim: int,
    seed: int,
) -> None:
    cpu_attention = tt.SelfAttention(
        embed_dim=embed_dim,
        rng=tt.Random(seed),
    )

    print(cpu_attention)

    cpu_parameters = cpu_attention.parameters()
    assert_parameter_layout(cpu_parameters, embed_dim)

    cpu_input = tt.Tensor(input_values)
    cpu_output = cpu_attention.forward(cpu_input)

    assert cpu_output.shape == list(input_values.shape)
    assert cpu_output.device == tt.Device.CPU

    cpu_grad_output = tt.Tensor(grad_output_values)
    cpu_grad_input = cpu_attention.backward(
        cpu_grad_output
    )

    assert cpu_grad_input.shape == list(input_values.shape)
    assert cpu_grad_input.device == tt.Device.CPU

    for parameter in cpu_parameters:
        assert np.all(np.isfinite(parameter.grad.numpy()))

    print("\nSelfAttention CPU output:")
    print(cpu_output.numpy())

    if tt.cuda_available():
        cuda_attention = tt.SelfAttention(
            embed_dim=embed_dim,
            rng=tt.Random(seed),
        )

        cuda_input = tt.Tensor(input_values)
        cuda_input.to_cuda()

        cuda_output = cuda_attention.forward(
            cuda_input
        )

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

        cuda_grad_input = cuda_attention.backward(
            cuda_grad_output
        )

        assert cuda_grad_input.device == tt.Device.CUDA

        np.testing.assert_allclose(
            cuda_grad_input.numpy(),
            cpu_grad_input.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

        cuda_parameters = cuda_attention.parameters()

        for cpu_parameter, cuda_parameter in zip(
            cpu_parameters,
            cuda_parameters,
        ):
            np.testing.assert_allclose(
                cuda_parameter.grad.numpy(),
                cpu_parameter.grad.numpy(),
                rtol=1e-3,
                atol=1e-3,
            )

        print("\nSelfAttention CUDA parity passed.")


def test_multi_head_attention(
    input_values: np.ndarray,
    grad_output_values: np.ndarray,
    embed_dim: int,
    num_heads: int,
    seed: int,
) -> None:
    config = tt.AttentionConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        causal=True,
        use_bias=True,
        attention_dropout=0.0,
        projection_dropout=0.0,
    )

    config.validate()

    assert config.head_dim == embed_dim // num_heads

    cpu_attention = tt.MultiHeadAttention(
        config=config,
        rng=tt.Random(seed),
    )

    print(cpu_attention)

    cpu_parameters = cpu_attention.parameters()
    assert_parameter_layout(cpu_parameters, embed_dim)

    cpu_input = tt.Tensor(input_values)
    cpu_output = cpu_attention.forward(cpu_input)

    assert cpu_output.shape == list(input_values.shape)
    assert cpu_output.device == tt.Device.CPU

    cpu_grad_output = tt.Tensor(
        grad_output_values
    )

    cpu_grad_input = cpu_attention.backward(
        cpu_grad_output
    )

    assert cpu_grad_input.shape == list(input_values.shape)
    assert cpu_grad_input.device == tt.Device.CPU

    for parameter in cpu_parameters:
        assert np.all(np.isfinite(parameter.grad.numpy()))

    print("\nMultiHeadAttention CPU output:")
    print(cpu_output.numpy())

    if tt.cuda_available():
        cuda_config = tt.AttentionConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            causal=True,
            use_bias=True,
            attention_dropout=0.0,
            projection_dropout=0.0,
        )

        cuda_attention = tt.MultiHeadAttention(
            config=cuda_config,
            rng=tt.Random(seed),
        )

        cuda_input = tt.Tensor(input_values)
        cuda_input.to_cuda()

        cuda_output = cuda_attention.forward(
            cuda_input
        )

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

        cuda_grad_input = cuda_attention.backward(
            cuda_grad_output
        )

        assert cuda_grad_input.device == tt.Device.CUDA

        np.testing.assert_allclose(
            cuda_grad_input.numpy(),
            cpu_grad_input.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

        cuda_parameters = cuda_attention.parameters()

        for cpu_parameter, cuda_parameter in zip(
            cpu_parameters,
            cuda_parameters,
        ):
            np.testing.assert_allclose(
                cuda_parameter.grad.numpy(),
                cpu_parameter.grad.numpy(),
                rtol=1e-3,
                atol=1e-3,
            )

        print("\nMultiHeadAttention CUDA parity passed.")


def test_failures(embed_dim: int) -> None:
    unused_self_attention = tt.SelfAttention(
        embed_dim=embed_dim,
        rng=tt.Random(99),
    )

    try:
        unused_self_attention.backward(
            tt.Tensor([1, 2, embed_dim], 1.0)
        )
        raise AssertionError(
            "Expected SelfAttention backward-before-forward to fail."
        )
    except RuntimeError:
        pass

    try:
        unused_self_attention.forward(
            tt.Tensor([2, embed_dim])
        )
        raise AssertionError(
            "Expected SelfAttention rank-2 input to fail."
        )
    except ValueError:
        pass

    try:
        unused_self_attention.forward(
            tt.Tensor([1, 2, embed_dim + 1])
        )
        raise AssertionError(
            "Expected SelfAttention feature mismatch to fail."
        )
    except ValueError:
        pass

    config = tt.AttentionConfig(
        embed_dim=embed_dim,
        num_heads=2,
        causal=True,
    )

    unused_multi_head = tt.MultiHeadAttention(
        config=config,
        rng=tt.Random(99),
    )

    try:
        unused_multi_head.backward(
            tt.Tensor([1, 2, embed_dim], 1.0)
        )
        raise AssertionError(
            "Expected MultiHeadAttention backward-before-forward to fail."
        )
    except RuntimeError:
        pass

    try:
        unused_multi_head.forward(
            tt.Tensor([2, embed_dim])
        )
        raise AssertionError(
            "Expected MultiHeadAttention rank-2 input to fail."
        )
    except ValueError:
        pass

    try:
        unused_multi_head.forward(
            tt.Tensor([1, 2, embed_dim + 1])
        )
        raise AssertionError(
            "Expected MultiHeadAttention feature mismatch to fail."
        )
    except ValueError:
        pass


def main() -> None:
    seed = 1234
    embed_dim = 4
    num_heads = 2

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

    test_self_attention(
        input_values,
        grad_output_values,
        embed_dim,
        seed,
    )

    test_multi_head_attention(
        input_values,
        grad_output_values,
        embed_dim,
        num_heads,
        seed,
    )

    test_failures(embed_dim)

    print("\nAttention tests passed.")


if __name__ == "__main__":
    main()