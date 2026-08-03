import numpy as np

import transformer_toy as tt


def main() -> None:
    # ----------------------------------------------------------
    # Default construction
    # ----------------------------------------------------------
    default_parameter = tt.Parameter()

    print("Default parameter:")
    print(f"  {default_parameter}")
    print(f"  name: {default_parameter.name!r}")
    print(f"  size: {default_parameter.size}")
    print(f"  requires_grad: {default_parameter.requires_grad}")
    print(f"  has_grad: {default_parameter.has_grad}")

    assert default_parameter.name == ""
    assert default_parameter.size == 0
    assert default_parameter.requires_grad is True
    assert default_parameter.has_grad is True
    assert default_parameter.value.empty is True
    assert default_parameter.grad.empty is True

    default_parameter.validate()

    # ----------------------------------------------------------
    # Parameterized construction
    # ----------------------------------------------------------
    source = tt.Tensor(
        np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )
    )

    parameter = tt.Parameter(
        value=source,
        name="linear.weight",
        requires_grad=True,
    )

    print("\nParameterized parameter:")
    print(f"  {parameter}")
    print(f"  value:\n{parameter.value.numpy()}")
    print(f"  grad:\n{parameter.grad.numpy()}")

    assert parameter.name == "linear.weight"
    assert parameter.requires_grad is True
    assert parameter.has_grad is True
    assert parameter.size == 6
    assert parameter.value.shape == [2, 3]
    assert parameter.grad.shape == [2, 3]

    np.testing.assert_allclose(
        parameter.value.numpy(),
        source.numpy(),
    )

    np.testing.assert_allclose(
        parameter.grad.numpy(),
        np.zeros((2, 3), dtype=np.float32),
    )

    parameter.validate()

    # ----------------------------------------------------------
    # Verify constructor copy semantics
    # ----------------------------------------------------------
    source.fill(99.0)

    np.testing.assert_allclose(
        parameter.value.numpy(),
        np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        ),
    )

    # ----------------------------------------------------------
    # Verify embedded Tensor reference behavior
    # ----------------------------------------------------------
    parameter.value.set_at([0, 0], 10.0)
    parameter.grad.fill(3.0)

    assert abs(parameter.value.at([0, 0]) - 10.0) < 1e-6

    np.testing.assert_allclose(
        parameter.grad.numpy(),
        np.full((2, 3), 3.0, dtype=np.float32),
    )

    # ----------------------------------------------------------
    # zero_grad
    # ----------------------------------------------------------
    parameter.zero_grad()

    np.testing.assert_allclose(
        parameter.grad.numpy(),
        np.zeros((2, 3), dtype=np.float32),
    )

    # zero_grad intentionally does nothing when gradients are disabled.
    parameter.grad.fill(7.0)
    parameter.requires_grad = False
    parameter.zero_grad()

    np.testing.assert_allclose(
        parameter.grad.numpy(),
        np.full((2, 3), 7.0, dtype=np.float32),
    )

    assert parameter.has_grad is False

    # ----------------------------------------------------------
    # Tensor replacement
    # ----------------------------------------------------------
    replacement_value = tt.Tensor([4], 2.5)
    replacement_grad = tt.Tensor([4], 0.5)

    parameter.value = replacement_value
    parameter.grad = replacement_grad
    parameter.name = "replacement"
    parameter.requires_grad = True

    assert parameter.name == "replacement"
    assert parameter.size == 4
    assert parameter.value.shape == [4]
    assert parameter.grad.shape == [4]

    parameter.validate()

    # ----------------------------------------------------------
    # Validation failure
    # ----------------------------------------------------------
    parameter.grad = tt.Tensor([3])

    try:
        parameter.validate()
        raise AssertionError(
            "Expected mismatched value and gradient sizes to fail."
        )
    except RuntimeError as error:
        assert "mismatched value/grad sizes" in str(error)

    print("\nParameter tests passed.")


if __name__ == "__main__":
    main()