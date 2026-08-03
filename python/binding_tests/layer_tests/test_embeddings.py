import numpy as np

import transformer_toy as tt


def main() -> None:
    seed = 1234
    vocab_size = 6
    embedding_dim = 4

    # ----------------------------------------------------------
    # Construction and parameter inspection
    # ----------------------------------------------------------
    cpu_embedding = tt.Embedding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rng=tt.Random(seed),
    )

    print(cpu_embedding)

    assert cpu_embedding.table.shape == [
        vocab_size,
        embedding_dim,
    ]

    parameters = cpu_embedding.parameters()

    assert len(parameters) == 1

    table_parameter = parameters[0]

    assert isinstance(table_parameter, tt.Parameter)
    assert table_parameter.name == "embedding.table"
    assert table_parameter.value.shape == [
        vocab_size,
        embedding_dim,
    ]
    assert table_parameter.grad.shape == [
        vocab_size,
        embedding_dim,
    ]

    np.testing.assert_allclose(
        table_parameter.grad.numpy(),
        np.zeros(
            (vocab_size, embedding_dim),
            dtype=np.float32,
        ),
    )

    # ----------------------------------------------------------
    # CPU forward
    # ----------------------------------------------------------
    token_ids_np = np.array(
        [
            [0, 2, 4],
            [1, 2, 5],
        ],
        dtype=np.float32,
    )

    token_ids = tt.Tensor(token_ids_np)
    output = cpu_embedding.forward(token_ids)

    print("\nCPU output:")
    print(output.numpy())

    assert output.shape == [2, 3, embedding_dim]
    assert output.device == tt.Device.CPU

    table_np = cpu_embedding.table.numpy()

    expected_output = table_np[
        token_ids_np.astype(np.int64)
    ]

    np.testing.assert_allclose(
        output.numpy(),
        expected_output,
        rtol=1e-5,
        atol=1e-6,
    )

    # ----------------------------------------------------------
    # CPU backward
    # ----------------------------------------------------------
    grad_output_np = np.arange(
        2 * 3 * embedding_dim,
        dtype=np.float32,
    ).reshape(2, 3, embedding_dim)

    grad_output = tt.Tensor(grad_output_np)
    grad_input = cpu_embedding.backward(grad_output)

    assert grad_input.shape == [2, 3]

    np.testing.assert_allclose(
        grad_input.numpy(),
        np.zeros((2, 3), dtype=np.float32),
    )

    expected_table_grad = np.zeros(
        (vocab_size, embedding_dim),
        dtype=np.float32,
    )

    for batch in range(token_ids_np.shape[0]):
        for position in range(token_ids_np.shape[1]):
            token_id = int(token_ids_np[batch, position])

            expected_table_grad[token_id] += (
                grad_output_np[batch, position]
            )

    np.testing.assert_allclose(
        table_parameter.grad.numpy(),
        expected_table_grad,
        rtol=1e-5,
        atol=1e-6,
    )

    # Token ID 2 occurs twice, so its gradient should be accumulated.
    expected_token_two_grad = (
        grad_output_np[0, 1] +
        grad_output_np[1, 1]
    )

    np.testing.assert_allclose(
        table_parameter.grad.numpy()[2],
        expected_token_two_grad,
        rtol=1e-5,
        atol=1e-6,
    )

    # ----------------------------------------------------------
    # CUDA parity
    # ----------------------------------------------------------
    if tt.cuda_available():
        cuda_embedding = tt.Embedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            rng=tt.Random(seed),
        )

        cuda_token_ids = tt.Tensor(token_ids_np)
        cuda_token_ids.to_cuda()

        cuda_output = cuda_embedding.forward(
            cuda_token_ids
        )

        assert cuda_output.device == tt.Device.CUDA

        np.testing.assert_allclose(
            cuda_output.numpy(),
            output.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        cuda_grad_output = tt.Tensor(
            grad_output_np
        )
        cuda_grad_output.to_cuda()

        cuda_grad_input = cuda_embedding.backward(
            cuda_grad_output
        )

        assert cuda_grad_input.device == tt.Device.CUDA

        np.testing.assert_allclose(
            cuda_grad_input.numpy(),
            grad_input.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        cuda_parameter = (
            cuda_embedding.parameters()[0]
        )

        np.testing.assert_allclose(
            cuda_parameter.grad.numpy(),
            table_parameter.grad.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        print("\nCUDA parity passed.")

    # ----------------------------------------------------------
    # Failure: backward before forward
    # ----------------------------------------------------------
    unused_embedding = tt.Embedding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rng=tt.Random(99),
    )

    try:
        unused_embedding.backward(
            tt.Tensor([1, 2, embedding_dim], 1.0)
        )
        raise AssertionError(
            "Expected backward-before-forward to fail."
        )
    except RuntimeError:
        pass

    # ----------------------------------------------------------
    # Failure: incorrect token input rank
    # ----------------------------------------------------------
    try:
        cpu_embedding.forward(
            tt.Tensor([2, 3, 1])
        )
        raise AssertionError(
            "Expected rank-3 token input to fail."
        )
    except ValueError:
        pass

    # ----------------------------------------------------------
    # Failure: token ID outside vocabulary
    # ----------------------------------------------------------
    try:
        cpu_embedding.forward(
            tt.Tensor.from_data(
                [0.0, float(vocab_size)],
                [1, 2],
            )
        )
        raise AssertionError(
            "Expected out-of-range token ID to fail."
        )
    except IndexError:
        pass

    # ----------------------------------------------------------
    # Failure: incorrect backward embedding dimension
    # ----------------------------------------------------------
    cpu_embedding.forward(token_ids)

    try:
        cpu_embedding.backward(
            tt.Tensor([2, 3, embedding_dim + 1])
        )
        raise AssertionError(
            "Expected embedding-dimension mismatch to fail."
        )
    except ValueError:
        pass

    print("\nEmbedding tests passed.")


if __name__ == "__main__":
    main()