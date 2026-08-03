import numpy as np

import transformer_toy as tt


def main() -> None:
    seed = 1234

    # ----------------------------------------------------------
    # Construction and parameter inspection
    # ----------------------------------------------------------
    cpu_layer = tt.Linear(
        in_features=3,
        out_features=2,
        rng=tt.Random(seed),
    )

    print(cpu_layer)

    assert cpu_layer.weights.shape == [2, 3]
    assert cpu_layer.bias.shape == [2]

    parameters = cpu_layer.parameters()

    assert len(parameters) == 2

    weight_parameter = parameters[0]
    bias_parameter = parameters[1]

    assert isinstance(weight_parameter, tt.Parameter)
    assert isinstance(bias_parameter, tt.Parameter)

    assert weight_parameter.name == "linear.weight"
    assert bias_parameter.name == "linear.bias"

    assert weight_parameter.value.shape == [2, 3]
    assert weight_parameter.grad.shape == [2, 3]

    assert bias_parameter.value.shape == [2]
    assert bias_parameter.grad.shape == [2]

    np.testing.assert_allclose(
        bias_parameter.value.numpy(),
        np.zeros(2, dtype=np.float32),
    )

    # ----------------------------------------------------------
    # CPU forward
    # ----------------------------------------------------------
    input_values = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )

    cpu_input = tt.Tensor(input_values)
    cpu_output = cpu_layer.forward(cpu_input)

    print("\nCPU output:")
    print(cpu_output.numpy())

    assert cpu_output.shape == [2, 2]
    assert cpu_output.device == tt.Device.CPU

    expected_output = (
        input_values @ cpu_layer.weights.numpy().T
        + cpu_layer.bias.numpy()
    )

    np.testing.assert_allclose(
        cpu_output.numpy(),
        expected_output,
        rtol=1e-5,
        atol=1e-6,
    )

    # ----------------------------------------------------------
    # CPU backward
    # ----------------------------------------------------------
    grad_output_values = np.array(
        [
            [1.0, 0.5],
            [-0.25, 2.0],
        ],
        dtype=np.float32,
    )

    cpu_grad_output = tt.Tensor(grad_output_values)
    cpu_grad_input = cpu_layer.backward(cpu_grad_output)

    print("\nCPU input gradient:")
    print(cpu_grad_input.numpy())

    assert cpu_grad_input.shape == [2, 3]

    expected_grad_input = (
        grad_output_values @ cpu_layer.weights.numpy()
    )

    np.testing.assert_allclose(
        cpu_grad_input.numpy(),
        expected_grad_input,
        rtol=1e-5,
        atol=1e-6,
    )

    expected_weight_grad = (
        grad_output_values.T @ input_values
    )

    expected_bias_grad = grad_output_values.sum(axis=0)

    np.testing.assert_allclose(
        weight_parameter.grad.numpy(),
        expected_weight_grad,
        rtol=1e-5,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        bias_parameter.grad.numpy(),
        expected_bias_grad,
        rtol=1e-5,
        atol=1e-6,
    )

    # ----------------------------------------------------------
    # CUDA parity
    # ----------------------------------------------------------
    if tt.cuda_available():
        cuda_layer = tt.Linear(
            in_features=3,
            out_features=2,
            rng=tt.Random(seed),
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

        cuda_grad_output = tt.Tensor(grad_output_values)
        cuda_grad_output.to_cuda()

        cuda_grad_input = cuda_layer.backward(
            cuda_grad_output
        )

        assert cuda_grad_input.device == tt.Device.CUDA

        np.testing.assert_allclose(
            cuda_grad_input.numpy(),
            cpu_grad_input.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        cuda_parameters = cuda_layer.parameters()

        np.testing.assert_allclose(
            cuda_parameters[0].grad.numpy(),
            weight_parameter.grad.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        np.testing.assert_allclose(
            cuda_parameters[1].grad.numpy(),
            bias_parameter.grad.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        print("\nCUDA parity passed.")

    # ----------------------------------------------------------
    # Failure: backward before forward
    # ----------------------------------------------------------
    unused_layer = tt.Linear(
        in_features=3,
        out_features=2,
        rng=tt.Random(99),
    )

    try:
        unused_layer.backward(
            tt.Tensor([1, 2], 1.0)
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
        cpu_layer.forward(
            tt.Tensor([2, 3, 1])
        )
        raise AssertionError(
            "Expected rank-3 input to fail."
        )
    except ValueError:
        pass

    # ----------------------------------------------------------
    # Failure: wrong input feature count
    # ----------------------------------------------------------
    try:
        cpu_layer.forward(
            tt.Tensor([2, 4])
        )
        raise AssertionError(
            "Expected feature mismatch to fail."
        )
    except ValueError:
        pass

    print("\nLinear tests passed.")


if __name__ == "__main__":
    main()