import numpy as np

import transformer_toy as tt


def make_parameter(
    values: np.ndarray,
    gradients: np.ndarray,
    name: str,
) -> tt.Parameter:
    parameter = tt.Parameter(
        value=tt.Tensor(
            values.astype(np.float32)
        ),
        name=name,
        requires_grad=True,
    )

    parameter.grad = tt.Tensor(
        gradients.astype(np.float32)
    )

    parameter.validate()

    return parameter


def test_sgd_cpu() -> None:
    initial_values = np.array(
        [1.0, -2.0, 3.0],
        dtype=np.float32,
    )

    gradients = np.array(
        [0.5, -1.0, 2.0],
        dtype=np.float32,
    )

    parameter = make_parameter(
        initial_values,
        gradients,
        "sgd.parameter",
    )

    optimizer = tt.SGDOptimizer(
        learning_rate=0.1,
        weight_decay=0.0,
    )

    print(optimizer)

    assert isinstance(optimizer, tt.Optimizer)
    assert abs(optimizer.learning_rate - 0.1) < 1e-6

    optimizer.step([parameter])

    expected = (
        initial_values
        - 0.1 * gradients
    )

    np.testing.assert_allclose(
        parameter.value.numpy(),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )

    optimizer.zero_grad([parameter])

    np.testing.assert_allclose(
        parameter.grad.numpy(),
        np.zeros_like(gradients),
    )

    optimizer.learning_rate = 0.05

    assert abs(
        optimizer.learning_rate - 0.05
    ) < 1e-6


def test_sgd_weight_decay() -> None:
    initial_values = np.array(
        [1.0, -2.0],
        dtype=np.float32,
    )

    gradients = np.array(
        [0.5, 0.25],
        dtype=np.float32,
    )

    parameter = make_parameter(
        initial_values,
        gradients,
        "sgd.weight_decay",
    )

    learning_rate = 0.1
    weight_decay = 0.2

    optimizer = tt.SGDOptimizer(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    optimizer.step([parameter])

    expected_gradient = (
        gradients
        + weight_decay * initial_values
    )

    expected_values = (
        initial_values
        - learning_rate * expected_gradient
    )

    np.testing.assert_allclose(
        parameter.value.numpy(),
        expected_values,
        rtol=1e-6,
        atol=1e-6,
    )


def test_adam_cpu() -> None:
    initial_values = np.array(
        [1.0, -2.0, 3.0],
        dtype=np.float32,
    )

    gradients = np.array(
        [0.5, -1.0, 2.0],
        dtype=np.float32,
    )

    parameter = make_parameter(
        initial_values,
        gradients,
        "adam.parameter",
    )

    learning_rate = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1.0e-8

    optimizer = tt.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=0.0,
    )

    print(optimizer)

    assert isinstance(optimizer, tt.Optimizer)

    optimizer.step([parameter])

    # On Adam's first step, bias correction produces:
    # m_hat = gradient
    # v_hat = gradient squared
    expected = (
        initial_values
        - learning_rate
        * gradients
        / (
            np.sqrt(gradients * gradients)
            + epsilon
        )
    )

    np.testing.assert_allclose(
        parameter.value.numpy(),
        expected,
        rtol=1e-5,
        atol=1e-6,
    )

    optimizer.zero_grad([parameter])

    np.testing.assert_allclose(
        parameter.grad.numpy(),
        np.zeros_like(gradients),
    )


def test_requires_grad_skip() -> None:
    parameter = make_parameter(
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([5.0, 5.0], dtype=np.float32),
        "frozen.parameter",
    )

    parameter.requires_grad = False

    before = parameter.value.numpy().copy()

    optimizer = tt.SGDOptimizer(
        learning_rate=0.1,
    )

    optimizer.step([parameter])
    optimizer.zero_grad([parameter])

    np.testing.assert_allclose(
        parameter.value.numpy(),
        before,
    )

    # zeroGrad intentionally does nothing when requires_grad is false.
    np.testing.assert_allclose(
        parameter.grad.numpy(),
        np.array([5.0, 5.0], dtype=np.float32),
    )


def test_cuda_parity() -> None:
    if not tt.cuda_available():
        return

    initial_values = np.array(
        [1.0, -2.0, 3.0, -4.0],
        dtype=np.float32,
    )

    gradients = np.array(
        [0.5, -1.0, 2.0, -0.25],
        dtype=np.float32,
    )

    cpu_sgd_parameter = make_parameter(
        initial_values,
        gradients,
        "cpu.sgd",
    )

    cuda_sgd_parameter = make_parameter(
        initial_values,
        gradients,
        "cuda.sgd",
    )

    cuda_sgd_parameter.value.to_cuda()
    cuda_sgd_parameter.grad.to_cuda()

    cpu_sgd = tt.SGDOptimizer(
        learning_rate=0.1,
        weight_decay=0.01,
    )

    cuda_sgd = tt.SGDOptimizer(
        learning_rate=0.1,
        weight_decay=0.01,
    )

    cpu_sgd.step([cpu_sgd_parameter])
    cuda_sgd.step([cuda_sgd_parameter])

    np.testing.assert_allclose(
        cuda_sgd_parameter.value.numpy(),
        cpu_sgd_parameter.value.numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    cpu_adam_parameter = make_parameter(
        initial_values,
        gradients,
        "cpu.adam",
    )

    cuda_adam_parameter = make_parameter(
        initial_values,
        gradients,
        "cuda.adam",
    )

    cuda_adam_parameter.value.to_cuda()
    cuda_adam_parameter.grad.to_cuda()

    cpu_adam = tt.AdamOptimizer(
        learning_rate=0.01,
    )

    cuda_adam = tt.AdamOptimizer(
        learning_rate=0.01,
    )

    cpu_adam.step([cpu_adam_parameter])
    cuda_adam.step([cuda_adam_parameter])

    np.testing.assert_allclose(
        cuda_adam_parameter.value.numpy(),
        cpu_adam_parameter.value.numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    print("CUDA optimizer parity passed.")


def test_failures() -> None:
    try:
        tt.SGDOptimizer(
            learning_rate=0.0,
        )

        raise AssertionError(
            "Expected invalid SGD learning rate to fail."
        )
    except ValueError:
        pass

    try:
        tt.AdamOptimizer(
            beta1=1.0,
        )

        raise AssertionError(
            "Expected invalid Adam beta1 to fail."
        )
    except ValueError:
        pass

    optimizer = tt.AdamOptimizer()

    try:
        optimizer.learning_rate = -0.1

        raise AssertionError(
            "Expected invalid learning-rate assignment to fail."
        )
    except ValueError:
        pass

    try:
        optimizer.step([object()])

        raise AssertionError(
            "Expected non-Parameter input to fail."
        )
    except TypeError:
        pass


def main() -> None:
    test_sgd_cpu()
    test_sgd_weight_decay()
    test_adam_cpu()
    test_requires_grad_skip()
    test_cuda_parity()
    test_failures()

    print("\nOptimizer tests passed.")


if __name__ == "__main__":
    main()