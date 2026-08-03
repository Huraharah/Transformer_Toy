import numpy as np

import transformer_toy as tt


def gelu(values: np.ndarray) -> np.ndarray:
    coefficient = np.float32(0.044715)
    scale = np.float32(0.7978845608028654)

    return (
        np.float32(0.5)
        * values
        * (
            np.float32(1.0)
            + np.tanh(
                scale
                * (
                    values
                    + coefficient * values**3
                )
            )
        )
    )


def main() -> None:
    seed = 1234
    embed_dim = 4
    hidden_dim = 6

    # ----------------------------------------------------------
    # Construction and parameters
    # ----------------------------------------------------------
    cpu_ffn = tt.FFN(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        rng=tt.Random(seed),
    )

    print(cpu_ffn)

    parameters = cpu_ffn.parameters()

    assert len(parameters) == 4

    linear1_weight = parameters[0]
    linear1_bias = parameters[1]
    linear2_weight = parameters[2]
    linear2_bias = parameters[3]

    assert linear1_weight.name == "linear.weight"
    assert linear1_bias.name == "linear.bias"
    assert linear2_weight.name == "linear.weight"
    assert linear2_bias.name == "linear.bias"

    assert linear1_weight.value.shape == [
        hidden_dim,
        embed_dim,
    ]
    assert linear1_bias.value.shape == [
        hidden_dim,
    ]
    assert linear2_weight.value.shape == [
        embed_dim,
        hidden_dim,
    ]
    assert linear2_bias.value.shape == [
        embed_dim,
    ]

    # ----------------------------------------------------------
    # CPU forward
    # ----------------------------------------------------------
    input_values = np.array(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [0.5, -1.0, 2.0, 1.5],
            ],
            [
                [-2.0, 0.0, 1.0, 3.0],
                [4.0, 3.0, 2.0, 1.0],
            ],
        ],
        dtype=np.float32,
    )

    cpu_input = tt.Tensor(input_values)
    cpu_output = cpu_ffn.forward(cpu_input)

    assert cpu_output.shape == [2, 2, embed_dim]
    assert cpu_output.device == tt.Device.CPU

    flat_input = input_values.reshape(-1, embed_dim)

    hidden = (
        flat_input
        @ linear1_weight.value.numpy().T
        + linear1_bias.value.numpy()
    )

    activated = gelu(hidden)

    expected_output = (
        activated
        @ linear2_weight.value.numpy().T
        + linear2_bias.value.numpy()
    ).reshape(2, 2, embed_dim)

    np.testing.assert_allclose(
        cpu_output.numpy(),
        expected_output,
        rtol=1e-5,
        atol=1e-5,
    )

    print("\nCPU output:")
    print(cpu_output.numpy())

    # ----------------------------------------------------------
    # CPU backward
    # ----------------------------------------------------------
    grad_output_values = np.arange(
        2 * 2 * embed_dim,
        dtype=np.float32,
    ).reshape(2, 2, embed_dim) / 10.0

    grad_output = tt.Tensor(grad_output_values)
    grad_input = cpu_ffn.backward(grad_output)

    assert grad_input.shape == [2, 2, embed_dim]
    assert grad_input.device == tt.Device.CPU

    for parameter in parameters:
        assert parameter.grad.shape == parameter.value.shape

    # Confirm gradients are populated.
    assert np.any(
        np.abs(linear1_weight.grad.numpy()) > 0.0
    )
    assert np.any(
        np.abs(linear2_weight.grad.numpy()) > 0.0
    )

    print("\nCPU input gradient:")
    print(grad_input.numpy())

    # ----------------------------------------------------------
    # CUDA parity
    # ----------------------------------------------------------
    if tt.cuda_available():
        cuda_ffn = tt.FFN(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            rng=tt.Random(seed),
        )

        cuda_input = tt.Tensor(input_values)
        cuda_input.to_cuda()

        cuda_output = cuda_ffn.forward(cuda_input)

        assert cuda_output.device == tt.Device.CUDA

        np.testing.assert_allclose(
            cuda_output.numpy(),
            cpu_output.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        cuda_grad_output = tt.Tensor(
            grad_output_values
        )
        cuda_grad_output.to_cuda()

        cuda_grad_input = cuda_ffn.backward(
            cuda_grad_output
        )

        assert cuda_grad_input.device == tt.Device.CUDA

        np.testing.assert_allclose(
            cuda_grad_input.numpy(),
            grad_input.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

        cuda_parameters = cuda_ffn.parameters()

        for cpu_parameter, cuda_parameter in zip(
            parameters,
            cuda_parameters,
        ):
            np.testing.assert_allclose(
                cuda_parameter.grad.numpy(),
                cpu_parameter.grad.numpy(),
                rtol=1e-4,
                atol=1e-4,
            )

        print("\nCUDA parity passed.")

    # ----------------------------------------------------------
    # Failure: backward before forward
    # ----------------------------------------------------------
    unused_ffn = tt.FFN(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        rng=tt.Random(99),
    )

    try:
        unused_ffn.backward(
            tt.Tensor([1, 2, embed_dim], 1.0)
        )
        raise AssertionError(
            "Expected backward-before-forward to fail."
        )
    except RuntimeError:
        pass

    # ----------------------------------------------------------
    # Failure: incorrect forward rank
    # ----------------------------------------------------------
    try:
        cpu_ffn.forward(
            tt.Tensor([2, embed_dim])
        )
        raise AssertionError(
            "Expected rank-2 input to fail."
        )
    except ValueError:
        pass

    # ----------------------------------------------------------
    # Failure: incorrect feature dimension
    # ----------------------------------------------------------
    try:
        cpu_ffn.forward(
            tt.Tensor([1, 2, embed_dim + 1])
        )
        raise AssertionError(
            "Expected feature mismatch to fail."
        )
    except ValueError:
        pass

    # ----------------------------------------------------------
    # Failure: backward batch/sequence mismatch
    # ----------------------------------------------------------
    cpu_ffn.forward(cpu_input)

    try:
        cpu_ffn.backward(
            tt.Tensor([1, 2, embed_dim], 1.0)
        )
        raise AssertionError(
            "Expected backward shape mismatch to fail."
        )
    except ValueError:
        pass

    print("\nFFN tests passed.")


if __name__ == "__main__":
    main()