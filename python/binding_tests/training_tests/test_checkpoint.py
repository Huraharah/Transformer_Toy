from pathlib import Path
import tempfile

import numpy as np

import transformer_toy as tt


def assert_close(actual: float, expected: float) -> None:
    assert abs(actual - expected) < 1e-6


def make_parameter(
    values: np.ndarray,
    gradients: np.ndarray,
    name: str,
    requires_grad: bool = True,
) -> tt.Parameter:
    parameter = tt.Parameter(
        value=tt.Tensor(values.astype(np.float32)),
        name=name,
        requires_grad=requires_grad,
    )

    parameter.grad = tt.Tensor(
        gradients.astype(np.float32)
    )

    parameter.validate()

    return parameter


def main() -> None:
    parameter_a = make_parameter(
        values=np.array(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=np.float32,
        ),
        gradients=np.array(
            [[0.1, 0.2], [0.3, 0.4]],
            dtype=np.float32,
        ),
        name="layer.weight",
        requires_grad=True,
    )

    parameter_b = make_parameter(
        values=np.array(
            [5.0, 6.0],
            dtype=np.float32,
        ),
        gradients=np.array(
            [0.5, 0.6],
            dtype=np.float32,
        ),
        name="layer.bias",
        requires_grad=False,
    )

    parameters = [
        parameter_a,
        parameter_b,
    ]

    original_values = [
        parameter.value.numpy().copy()
        for parameter in parameters
    ]

    original_gradients = [
        parameter.grad.numpy().copy()
        for parameter in parameters
    ]

    metadata = tt.CheckpointMetadata()
    metadata.epoch = 7
    metadata.global_step = 1234
    metadata.run_name = "python_checkpoint_test"

    print(metadata)

    history = tt.TrainingHistory()

    history.add_train_loss(3.0)
    history.add_train_loss(2.5)
    history.add_train_loss(2.0)

    history.add_validation_loss(3.2)
    history.add_validation_loss(2.7)

    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path = (
            Path(temporary_directory)
            / "test_checkpoint.bin"
        )

        # ------------------------------------------------------
        # Save
        # ------------------------------------------------------
        tt.Checkpoint.save(
            path=str(checkpoint_path),
            parameters=parameters,
            metadata=metadata,
            history=history,
        )

        assert checkpoint_path.exists()
        assert checkpoint_path.stat().st_size > 0

        # ------------------------------------------------------
        # Corrupt all current in-memory state
        # ------------------------------------------------------
        parameter_a.value.fill(99.0)
        parameter_a.grad.fill(88.0)
        parameter_a.name = "corrupted.weight"
        parameter_a.requires_grad = False

        parameter_b.value.fill(-99.0)
        parameter_b.grad.fill(-88.0)
        parameter_b.name = "corrupted.bias"
        parameter_b.requires_grad = True

        metadata.epoch = -1
        metadata.global_step = -1
        metadata.run_name = "corrupted"

        history.clear()
        history.add_train_loss(999.0)
        history.add_validation_loss(888.0)

        # ------------------------------------------------------
        # Load
        # ------------------------------------------------------
        tt.Checkpoint.load(
            path=str(checkpoint_path),
            parameters=parameters,
            metadata=metadata,
            history=history,
        )

        # Metadata restored
        assert metadata.epoch == 7
        assert metadata.global_step == 1234
        assert metadata.run_name == "python_checkpoint_test"

        # Parameter metadata restored
        assert parameter_a.name == "layer.weight"
        assert parameter_a.requires_grad is True

        assert parameter_b.name == "layer.bias"
        assert parameter_b.requires_grad is False

        # Parameter values and gradients restored
        for index, parameter in enumerate(parameters):
            np.testing.assert_allclose(
                parameter.value.numpy(),
                original_values[index],
                rtol=1e-6,
                atol=1e-6,
            )

            np.testing.assert_allclose(
                parameter.grad.numpy(),
                original_gradients[index],
                rtol=1e-6,
                atol=1e-6,
            )

            parameter.validate()

        # Training history restored
        np.testing.assert_allclose(
            history.train_losses,
            [3.0, 2.5, 2.0],
            rtol=1e-6,
            atol=1e-6,
        )

        np.testing.assert_allclose(
            history.validation_losses,
            [3.2, 2.7],
            rtol=1e-6,
            atol=1e-6,
        )

        assert_close(
            history.latest_train_loss,
            2.0,
        )

        assert_close(
            history.latest_validation_loss,
            2.7,
        )

        # ------------------------------------------------------
        # CUDA save synchronization
        # ------------------------------------------------------
        if tt.cuda_available():
            cuda_parameter = make_parameter(
                values=np.array(
                    [1.0, 2.0, 3.0],
                    dtype=np.float32,
                ),
                gradients=np.array(
                    [0.1, 0.2, 0.3],
                    dtype=np.float32,
                ),
                name="cuda.parameter",
            )

            cuda_parameter.value.to_cuda()
            cuda_parameter.grad.to_cuda()

            cuda_path = (
                Path(temporary_directory)
                / "cuda_checkpoint.bin"
            )

            cuda_metadata = tt.CheckpointMetadata()
            cuda_history = tt.TrainingHistory()

            tt.Checkpoint.save(
                str(cuda_path),
                [cuda_parameter],
                cuda_metadata,
                cuda_history,
            )

            cuda_parameter.value.fill(0.0)
            cuda_parameter.grad.fill(0.0)

            tt.Checkpoint.load(
                str(cuda_path),
                [cuda_parameter],
                cuda_metadata,
                cuda_history,
            )

            np.testing.assert_allclose(
                cuda_parameter.value.numpy(),
                [1.0, 2.0, 3.0],
                rtol=1e-6,
                atol=1e-6,
            )

            np.testing.assert_allclose(
                cuda_parameter.grad.numpy(),
                [0.1, 0.2, 0.3],
                rtol=1e-6,
                atol=1e-6,
            )

        # ------------------------------------------------------
        # Failure: parameter-count mismatch
        # ------------------------------------------------------
        try:
            tt.Checkpoint.load(
                str(checkpoint_path),
                [parameter_a],
                tt.CheckpointMetadata(),
                tt.TrainingHistory(),
            )

            raise AssertionError(
                "Expected parameter-count mismatch to fail."
            )
        except RuntimeError as error:
            assert "parameter count" in str(error).lower()

        # ------------------------------------------------------
        # Failure: missing file
        # ------------------------------------------------------
        try:
            tt.Checkpoint.load(
                str(
                    Path(temporary_directory)
                    / "missing.bin"
                ),
                parameters,
                tt.CheckpointMetadata(),
                tt.TrainingHistory(),
            )

            raise AssertionError(
                "Expected missing checkpoint to fail."
            )
        except RuntimeError as error:
            assert "failed to open" in str(error).lower()

        # ------------------------------------------------------
        # Failure: invalid checkpoint format
        # ------------------------------------------------------
        invalid_path = (
            Path(temporary_directory)
            / "invalid.bin"
        )

        invalid_path.write_bytes(
            b"this is not a transformer checkpoint"
        )

        try:
            tt.Checkpoint.load(
                str(invalid_path),
                parameters,
                tt.CheckpointMetadata(),
                tt.TrainingHistory(),
            )

            raise AssertionError(
                "Expected invalid checkpoint format to fail."
            )
        except RuntimeError:
            pass

    print("\nCheckpoint tests passed.")


if __name__ == "__main__":
    main()