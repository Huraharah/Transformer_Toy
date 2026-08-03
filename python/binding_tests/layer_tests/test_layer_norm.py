import numpy as np

import transformer_toy as tt


def layer_norm_reference(
    values: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    mean = values.mean(axis=-1, keepdims=True)
    variance = values.var(axis=-1, keepdims=True)

    normalized = (
        values - mean
    ) / np.sqrt(variance + epsilon)

    return normalized * gamma + beta


def main() -> None:
    feature_dim = 4
    epsilon = 1.0e-5

    # ----------------------------------------------------------
    # Construction
    # ----------------------------------------------------------
    cpu_layer = tt.LayerNorm(
        feature_dim=feature_dim,
        epsilon=epsilon,
    )

    print(cpu_layer)

    parameters = cpu_layer.parameters()

    assert len(parameters) == 2

    gamma = parameters[0]
    beta = parameters[1]

    assert gamma.name == "layernorm.gamma"
    assert beta.name == "layernorm.beta"

    assert gamma.value.shape == [feature_dim]
    assert beta.value.shape == [feature_dim]

    np.testing.assert_allclose(
        gamma.value.numpy(),
        np.ones(feature_dim, dtype=np.float32),
    )

    np.testing.assert_allclose(
        beta.value.numpy(),
        np.zeros(feature_dim, dtype=np.float32),
    )

    # ----------------------------------------------------------
    # CPU forward
    # ----------------------------------------------------------
    input_values = np.array(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 4.0, 6.0, 8.0],
            ],
            [
                [-1.0, 0.0, 1.0, 2.0],
                [5.0, 5.0, 5.0, 5.0],
            ],
        ],
        dtype=np.float32,
    )

    cpu_input = tt.Tensor(input_values)
    cpu_output = cpu_layer.forward(cpu_input)

    assert cpu_output.shape == [2, 2, feature_dim]
    assert cpu_output.device == tt.Device.CPU

    expected_output = layer_norm_reference(
        input_values,
        gamma.value.numpy(),
        beta.value.numpy(),
        epsilon,
    )

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
    grad_output_values = np.array(
        [
            [
                [1.0, 0.5, -0.5, 2.0],
                [0.25, 1.0, 1.5, -1.0],
            ],
            [
                [2.0, -1.0, 0.5, 0.25],
                [1.0, 1.0, 1.0, 1.0],
            ],
        ],
        dtype=np.float32,
    )

    grad_output = tt.Tensor(grad_output_values)
    grad_input = cpu_layer.backward(grad_output)

    assert grad_input.shape == list(input_values.shape)

    # Gamma and beta reference gradients
    mean = input_values.mean(axis=-1, keepdims=True)
    variance = input_values.var(axis=-1, keepdims=True)

    x_hat = (
        input_values - mean
    ) / np.sqrt(variance + epsilon)

    expected_gamma_grad = (
        grad_output_values * x_hat
    ).sum(axis=(0, 1))

    expected_beta_grad = (
        grad_output_values
    ).sum(axis=(0, 1))

    np.testing.assert_allclose(
        gamma.grad.numpy(),
        expected_gamma_grad,
        rtol=1e-5,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        beta.grad.numpy(),
        expected_beta_grad,
        rtol=1e-5,
        atol=1e-5,
    )

    # For each normalized row, the input gradient should sum
    # approximately to zero.
    np.testing.assert_allclose(
        grad_input.numpy().sum(axis=-1),
        np.zeros((2, 2), dtype=np.float32),
        atol=1e-5,
    )

    print("\nCPU input gradient:")
    print(grad_input.numpy())

    # ----------------------------------------------------------
    # CUDA parity
    # ----------------------------------------------------------
    if tt.cuda_available():
        cuda_layer = tt.LayerNorm(
            feature_dim=feature_dim,
            epsilon=epsilon,
        )

        cuda_input = tt.Tensor(input_values)
        cuda_input.to_cuda()

        cuda_output = cuda_layer.forward(cuda_input)

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

        cuda_grad_input = cuda_layer.backward(
            cuda_grad_output
        )

        assert cuda_grad_input.device == tt.Device.CUDA

        np.testing.assert_allclose(
            cuda_grad_input.numpy(),
            grad_input.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

        cuda_parameters = cuda_layer.parameters()

        np.testing.assert_allclose(
            cuda_parameters[0].grad.numpy(),
            gamma.grad.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

        np.testing.assert_allclose(
            cuda_parameters[1].grad.numpy(),
            beta.grad.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

        print("\nCUDA parity passed.")

    # ----------------------------------------------------------
    # Failure: backward before forward
    # ----------------------------------------------------------
    unused_layer = tt.LayerNorm(feature_dim)

    try:
        unused_layer.backward(
            tt.Tensor([1, 1, feature_dim], 1.0)
        )
        raise AssertionError(
            "Expected backward-before-forward to fail."
        )
    except RuntimeError:
        pass

    # ----------------------------------------------------------
    # Failure: incorrect input rank
    # ----------------------------------------------------------
    try:
        cpu_layer.forward(
            tt.Tensor([2, feature_dim])
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
        cpu_layer.forward(
            tt.Tensor([1, 2, feature_dim + 1])
        )
        raise AssertionError(
            "Expected feature mismatch to fail."
        )
    except ValueError:
        pass

    print("\nLayerNorm tests passed.")


if __name__ == "__main__":
    main()