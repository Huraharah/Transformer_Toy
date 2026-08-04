import numpy as np

import transformer_toy as tt


def reference_cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, np.ndarray]:
    shifted = logits - logits.max(
        axis=1,
        keepdims=True,
    )

    exponentials = np.exp(shifted)

    probabilities = (
        exponentials
        / exponentials.sum(
            axis=1,
            keepdims=True,
        )
    )

    batch_size = logits.shape[0]

    loss = -np.log(
        probabilities[
            np.arange(batch_size),
            targets,
        ]
    ).mean()

    gradient = probabilities.copy()

    gradient[
        np.arange(batch_size),
        targets,
    ] -= 1.0

    gradient /= batch_size

    return float(loss), gradient.astype(np.float32)


def main() -> None:
    logits_values = np.array(
        [
            [2.0, 1.0, 0.1],
            [0.5, 2.5, 1.0],
            [-1.0, 0.0, 3.0],
            [1.2, 0.7, -0.5],
        ],
        dtype=np.float32,
    )

    targets_values = np.array(
        [0, 1, 2, 1],
        dtype=np.float32,
    )

    expected_loss, expected_gradient = (
        reference_cross_entropy(
            logits_values,
            targets_values.astype(np.int64),
        )
    )

    # ----------------------------------------------------------
    # Construction and inheritance
    # ----------------------------------------------------------
    loss_function = tt.CrossEntropyLoss()

    print(loss_function)

    assert isinstance(
        loss_function,
        tt.CrossEntropyLoss,
    )
    assert isinstance(
        loss_function,
        tt.Loss,
    )

    # ----------------------------------------------------------
    # CPU forward
    # ----------------------------------------------------------
    logits = tt.Tensor(logits_values)
    targets = tt.Tensor(targets_values)

    loss_value = loss_function.forward(
        logits,
        targets,
    )

    print(f"\nCPU loss: {loss_value}")

    assert abs(loss_value - expected_loss) < 1e-6

    # ----------------------------------------------------------
    # CPU backward
    # ----------------------------------------------------------
    gradient = loss_function.backward()

    assert gradient.shape == [4, 3]
    assert gradient.device == tt.Device.CPU

    np.testing.assert_allclose(
        gradient.numpy(),
        expected_gradient,
        rtol=1e-5,
        atol=1e-6,
    )

    # Each row of softmax-cross-entropy gradient should sum to zero.
    np.testing.assert_allclose(
        gradient.numpy().sum(axis=1),
        np.zeros(4, dtype=np.float32),
        atol=1e-6,
    )

    print("\nCPU gradient:")
    print(gradient.numpy())

    # ----------------------------------------------------------
    # Verify backward returns cached gradient by value
    # ----------------------------------------------------------
    first_gradient = loss_function.backward()
    first_gradient.fill(99.0)

    second_gradient = loss_function.backward()

    np.testing.assert_allclose(
        second_gradient.numpy(),
        expected_gradient,
        rtol=1e-5,
        atol=1e-6,
    )

    # ----------------------------------------------------------
    # CUDA parity
    # ----------------------------------------------------------
    if tt.cuda_available():
        cuda_loss_function = tt.CrossEntropyLoss()

        cuda_logits = tt.Tensor(logits_values)
        cuda_targets = tt.Tensor(targets_values)

        cuda_logits.to_cuda()
        cuda_targets.to_cuda()

        cuda_loss_value = cuda_loss_function.forward(
            cuda_logits,
            cuda_targets,
        )

        assert abs(
            cuda_loss_value - loss_value
        ) < 1e-5

        cuda_gradient = cuda_loss_function.backward()

        assert cuda_gradient.device == tt.Device.CUDA

        np.testing.assert_allclose(
            cuda_gradient.numpy(),
            gradient.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        print("\nCUDA parity passed.")

    # ----------------------------------------------------------
    # Failure: logits must be rank 2
    # ----------------------------------------------------------
    try:
        loss_function.forward(
            tt.Tensor([2, 3, 4]),
            tt.Tensor([2]),
        )

        raise AssertionError(
            "Expected rank-3 logits to fail."
        )
    except ValueError:
        pass

    # ----------------------------------------------------------
    # Failure: targets must be rank 1
    # ----------------------------------------------------------
    try:
        loss_function.forward(
            tt.Tensor([2, 3]),
            tt.Tensor([2, 1]),
        )

        raise AssertionError(
            "Expected rank-2 targets to fail."
        )
    except ValueError:
        pass

    # ----------------------------------------------------------
    # Failure: batch sizes must match
    # ----------------------------------------------------------
    try:
        loss_function.forward(
            tt.Tensor([3, 4]),
            tt.Tensor([2]),
        )

        raise AssertionError(
            "Expected batch mismatch to fail."
        )
    except ValueError:
        pass

    # ----------------------------------------------------------
    # Failure: target class must be in range
    # ----------------------------------------------------------
    try:
        loss_function.forward(
            tt.Tensor.from_data(
                [
                    1.0, 2.0, 3.0,
                    3.0, 2.0, 1.0,
                ],
                [2, 3],
            ),
            tt.Tensor.from_data(
                [0.0, 3.0],
                [2],
            ),
        )

        raise AssertionError(
            "Expected target-class range failure."
        )
    except IndexError:
        pass

    print("\nCrossEntropyLoss tests passed.")


if __name__ == "__main__":
    main()