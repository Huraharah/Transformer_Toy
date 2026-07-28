import numpy as np

import transformer_toy as tt


def main() -> None:
    # ----------------------------------------------------------
    # Default construction
    # ----------------------------------------------------------
    empty = tt.Tensor()

    assert empty.empty is True
    assert empty.size == 0
    assert empty.rank == 0
    assert empty.shape == []

    # ----------------------------------------------------------
    # Shape construction
    # ----------------------------------------------------------
    zeros = tt.Tensor([2, 3])

    assert zeros.shape == [2, 3]
    assert zeros.strides == [3, 1]
    assert zeros.rank == 2
    assert zeros.size == 6
    assert zeros.empty is False
    assert zeros.device == tt.Device.CPU

    np.testing.assert_array_equal(
        zeros.numpy(),
        np.zeros((2, 3), dtype=np.float32),
    )

    # ----------------------------------------------------------
    # Fill construction
    # ----------------------------------------------------------
    filled = tt.Tensor([2, 2], 1.5)

    np.testing.assert_allclose(
        filled.numpy(),
        np.full((2, 2), 1.5, dtype=np.float32),
    )

    # ----------------------------------------------------------
    # Raw flat Python data
    # ----------------------------------------------------------
    raw = tt.Tensor.from_data(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2, 3],
    )

    np.testing.assert_allclose(
        raw.numpy(),
        np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        ),
    )

    # ----------------------------------------------------------
    # NumPy float32
    # ----------------------------------------------------------
    source32 = np.arange(
        12,
        dtype=np.float32,
    ).reshape(3, 4)

    tensor32 = tt.Tensor(source32)

    assert tensor32.shape == [3, 4]

    np.testing.assert_array_equal(
        tensor32.numpy(),
        source32,
    )

    # Verify copy-in semantics
    source32[0, 0] = 999.0
    assert tensor32.at([0, 0]) == 0.0

    # ----------------------------------------------------------
    # NumPy dtype conversion
    # ----------------------------------------------------------
    source64 = np.array(
        [[1.25, 2.5], [3.75, 5.0]],
        dtype=np.float64,
    )

    tensor64 = tt.Tensor(source64)
    result64 = tensor64.numpy()

    assert result64.dtype == np.float32

    np.testing.assert_allclose(
        result64,
        source64.astype(np.float32),
    )

    # ----------------------------------------------------------
    # Noncontiguous NumPy input
    # ----------------------------------------------------------
    source = np.arange(
        24,
        dtype=np.float32,
    ).reshape(4, 6)

    noncontiguous = source[:, ::2]

    assert noncontiguous.flags.c_contiguous is False

    noncontiguous_tensor = tt.Tensor(noncontiguous)

    np.testing.assert_array_equal(
        noncontiguous_tensor.numpy(),
        noncontiguous,
    )

    # ----------------------------------------------------------
    # Flat and multidimensional mutation
    # ----------------------------------------------------------
    raw[0] = 10.0
    raw.set_at([1, 2], 20.0)

    assert raw[0] == 10.0
    assert raw.at([1, 2]) == 20.0

    # ----------------------------------------------------------
    # Fill and reshape
    # ----------------------------------------------------------
    raw.fill(3.0)
    raw.reshape([3, 2])

    assert raw.shape == [3, 2]
    assert raw.strides == [2, 1]

    np.testing.assert_allclose(
        raw.numpy(),
        np.full((3, 2), 3.0, dtype=np.float32),
    )

    # ----------------------------------------------------------
    # NumPy protocol
    # ----------------------------------------------------------
    protocol_array = np.asarray(raw)

    assert protocol_array.dtype == np.float32

    np.testing.assert_allclose(
        protocol_array,
        raw.numpy(),
    )

    # Verify copy-out semantics
    protocol_array[0, 0] = 50.0
    assert raw.at([0, 0]) == 3.0

    # ----------------------------------------------------------
    # CUDA round trip
    # ----------------------------------------------------------
    if tt.cuda_available():
        cuda_tensor = tt.Tensor.from_data(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
        )

        cuda_tensor.to_cuda()

        assert cuda_tensor.device == tt.Device.CUDA
        assert cuda_tensor.has_device_data is True

        cuda_result = cuda_tensor.numpy()

        np.testing.assert_allclose(
            cuda_result,
            np.array(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=np.float32,
            ),
        )

    # ----------------------------------------------------------
    # Failure cases
    # ----------------------------------------------------------
    try:
        tt.Tensor.from_data(
            [1.0, 2.0, 3.0],
            [2, 2],
        )
        raise AssertionError(
            "Expected shape/data mismatch to fail."
        )
    except ValueError:
        pass

    try:
        raw.reshape([5, 5])
        raise AssertionError(
            "Expected invalid reshape to fail."
        )
    except ValueError:
        pass

    try:
        _ = raw[100]
        raise AssertionError(
            "Expected flat index failure."
        )
    except IndexError:
        pass

    print("Tensor tests passed.")


if __name__ == "__main__":
    main()