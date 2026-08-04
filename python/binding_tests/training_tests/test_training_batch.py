import numpy as np

import transformer_toy as tt


def main() -> None:
    # ----------------------------------------------------------
    # Default construction
    # ----------------------------------------------------------
    empty_batch = tt.TrainingBatch()

    print(empty_batch)

    assert empty_batch.inputs.empty is True
    assert empty_batch.targets.empty is True

    # ----------------------------------------------------------
    # Parameterized construction
    # ----------------------------------------------------------
    inputs_np = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )

    targets_np = np.array(
        [
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0],
        ],
        dtype=np.float32,
    )

    inputs = tt.Tensor(inputs_np)
    targets = tt.Tensor(targets_np)

    batch = tt.TrainingBatch(
        inputs=inputs,
        targets=targets,
    )

    print(batch)

    assert batch.inputs.shape == [2, 3]
    assert batch.targets.shape == [2, 3]

    np.testing.assert_allclose(
        batch.inputs.numpy(),
        inputs_np,
    )

    np.testing.assert_allclose(
        batch.targets.numpy(),
        targets_np,
    )

    # ----------------------------------------------------------
    # Constructor copy semantics
    # ----------------------------------------------------------
    inputs.fill(99.0)
    targets.fill(88.0)

    np.testing.assert_allclose(
        batch.inputs.numpy(),
        inputs_np,
    )

    np.testing.assert_allclose(
        batch.targets.numpy(),
        targets_np,
    )

    # ----------------------------------------------------------
    # Embedded reference behavior
    # ----------------------------------------------------------
    batch.inputs.set_at([0, 0], 10.0)
    batch.targets.set_at([1, 2], 20.0)

    assert abs(batch.inputs.at([0, 0]) - 10.0) < 1e-6
    assert abs(batch.targets.at([1, 2]) - 20.0) < 1e-6

    # ----------------------------------------------------------
    # Tensor replacement
    # ----------------------------------------------------------
    replacement_inputs = tt.Tensor(
        np.ones((3, 2), dtype=np.float32)
    )

    replacement_targets = tt.Tensor(
        np.zeros((3, 2), dtype=np.float32)
    )

    batch.inputs = replacement_inputs
    batch.targets = replacement_targets

    assert batch.inputs.shape == [3, 2]
    assert batch.targets.shape == [3, 2]

    np.testing.assert_allclose(
        batch.inputs.numpy(),
        np.ones((3, 2), dtype=np.float32),
    )

    np.testing.assert_allclose(
        batch.targets.numpy(),
        np.zeros((3, 2), dtype=np.float32),
    )

    print("\nTrainingBatch tests passed.")


if __name__ == "__main__":
    main()