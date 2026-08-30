"""Test the backward FP4 quantization module."""
import numpy as np
from tensor_bridge_fp4 import (
    quantize_forward, backward_quantize, rel_frobenius, FP4_GRID, FP4_GRID_SIGNED,
)


def test_grid():
    assert np.allclose(FP4_GRID, [0, 0.5, 1, 1.5, 2, 3, 4, 6]), "FP4 grid wrong"
    assert len(FP4_GRID_SIGNED) == 15, "signed grid should have 15 values"


def test_quantize_forward_roundtrip():
    x = np.array([[1.0, -1.0, 0.5, 0.0]])
    q, s = quantize_forward(x)
    # 1.0 -> 1.0, -1.0 -> -1.0, 0.5 -> 0.5, 0.0 -> 0.0
    assert np.allclose(q[0], [1.0, -1.0, 0.5, 0.0], atol=1e-6)


def test_backward_beats_naive():
    np.random.seed(1)
    M, K, N = 32, 32, 32
    A = np.random.normal(0, 0.2, (M, K))
    B = np.random.normal(0, 0.5, (K, N))
    C_ref = A @ B

    A_q, _ = quantize_forward(A)
    err_naive = rel_frobenius(A_q @ B, C_ref)
    A_opt = backward_quantize(A, B, iters=2)
    err_back = rel_frobenius(A_opt @ B, C_ref)

    assert err_back < err_naive, f"backward {err_back:.2f}% not better than naive {err_naive:.2f}%"
    print(f"  OK: naive={err_naive:.2f}% backward={err_back:.2f}% ({err_naive/err_back:.1f}x)")


def test_encodable():
    np.random.seed(1)
    A = np.random.normal(0, 0.2, (16, 16))
    B = np.random.normal(0, 0.5, (16, 16))
    _, scale = quantize_forward(A)
    A_opt = backward_quantize(A, B, iters=1)
    units = np.round(A_opt / scale, 6)
    assert np.all(np.isin(units, FP4_GRID_SIGNED)), "backward output not FP4-encodable"


if __name__ == "__main__":
    test_grid()
    test_quantize_forward_roundtrip()
    test_backward_beats_naive()
    test_encodable()
    print("All tests passed.")
