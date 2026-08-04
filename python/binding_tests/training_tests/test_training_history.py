from pathlib import Path
import csv
import tempfile
import numpy as np

import transformer_toy as tt


def assert_close(
    actual: float,
    expected: float,
) -> None:
    assert abs(actual - expected) < 1e-6


def main() -> None:
    history = tt.TrainingHistory()

    print(history)

    assert history.train_losses == []
    assert history.validation_losses == []
    assert history.train_count == 0
    assert history.validation_count == 0
    assert len(history) == 0

    # ----------------------------------------------------------
    # Empty-history failures
    # ----------------------------------------------------------
    try:
        _ = history.latest_train_loss
        raise AssertionError(
            "Expected missing training loss to fail."
        )
    except RuntimeError as error:
        assert "No training losses recorded" in str(error)

    try:
        _ = history.latest_validation_loss
        raise AssertionError(
            "Expected missing validation loss to fail."
        )
    except RuntimeError as error:
        assert "No validation losses recorded" in str(error)

    # ----------------------------------------------------------
    # Append losses
    # ----------------------------------------------------------
    history.add_train_loss(3.0)
    history.add_train_loss(2.5)
    history.add_train_loss(2.0)

    history.add_validation_loss(3.2)
    history.add_validation_loss(2.7)

    print(history)

    assert history.train_count == 3
    assert history.validation_count == 2
    assert len(history) == 3

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

    # ----------------------------------------------------------
    # Full-list replacement
    # ----------------------------------------------------------
    history.train_losses = [
        1.5,
        1.25,
        1.0,
    ]

    history.validation_losses = [
        1.6,
        1.4,
    ]

    np.testing.assert_allclose(
        history.train_losses,
        [1.5, 1.25, 1.0],
        rtol=1e-6,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        history.validation_losses,
        [1.6, 1.4],
        rtol=1e-6,
        atol=1e-6,
    )

    # Verify the getter returns a detached Python list.
    copied_losses = history.train_losses
    copied_losses.append(999.0)

    np.testing.assert_allclose(
        history.train_losses,
        [1.5, 1.25, 1.0],
        rtol=1e-6,
        atol=1e-6,
    )

    # ----------------------------------------------------------
    # CSV output
    # ----------------------------------------------------------
    with tempfile.TemporaryDirectory() as temporary_directory:
        csv_path = (
            Path(temporary_directory)
            / "training_history.csv"
        )

        history.save_csv(str(csv_path))

        assert csv_path.exists()

        with csv_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            rows = list(csv.reader(file))

        assert rows[0] == [
            "step",
            "train_loss",
            "validation_loss",
        ]

        assert rows[1][0] == "0"
        assert_close(float(rows[1][1]), 1.5)
        assert_close(float(rows[1][2]), 1.6)

        assert rows[2][0] == "1"
        assert_close(float(rows[2][1]), 1.25)
        assert_close(float(rows[2][2]), 1.4)

        assert rows[3][0] == "2"
        assert_close(float(rows[3][1]), 1.0)
        assert rows[3][2] == ""

    # ----------------------------------------------------------
    # Clear
    # ----------------------------------------------------------
    history.clear()

    assert history.train_losses == []
    assert history.validation_losses == []
    assert history.train_count == 0
    assert history.validation_count == 0
    assert len(history) == 0

    print("\nTrainingHistory tests passed.")


if __name__ == "__main__":
    main()