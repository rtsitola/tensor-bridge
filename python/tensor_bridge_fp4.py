"""
tensor_bridge_fp4 / fp4_backward.py — Backward FP4 quantization.

Optimize the FP4 (E2M1) representation of a weight matrix to minimize the *output*
error of a GEMM, instead of quantizing each value to the nearest grid point.

    forward (naive):   A --nearest-grid--> A_q       minimize ||A_q - A||  (wrong objective)
    backward (ours):   A, B, C=A@B -> find A_q in the FP4 grid minimizing ||A_q @ B - C||

This is the core idea of LLM-FP4 (EMNLP 2023), GPTQ, AWQ and AdaRound: minimize
output fidelity, not weight fidelity. The result stays FP4-encodable (every element
is `grid_value * scale`).

Grid: OCP E2M1 = {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}. Max magnitude 6.
"""

from __future__ import annotations
import numpy as np

# OCP E2M1 grid (15 distinct values; 0x0000 and 0x1000 both decode to +0)
FP4_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_GRID_SIGNED = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                            -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def quantize_forward(x: np.ndarray, scale: float | None = None) -> tuple[np.ndarray, float]:
    """Naive per-tensor FP4 quantization (nearest grid point). Returns (A_q, scale)."""
    if scale is None:
        scale = float(np.max(np.abs(x)) / 6.0)
        scale = scale if scale > 0 else 1.0
    xs = x / scale
    # quantize |xs| to the positive grid, then re-apply the sign of x
    idx = np.argmin(np.abs(np.abs(xs)[..., None] - FP4_GRID[None, :]), axis=-1)
    q = FP4_GRID[idx]
    q = np.where(xs < 0, -q, q)  # apply sign from x
    return q * scale, scale


def _row_err(a: np.ndarray, B: np.ndarray, c_target: np.ndarray) -> float:
    """|| a @ B - c_target ||_2 for one output row."""
    return float(np.linalg.norm(a @ B - c_target))


def optimize_row(a0: np.ndarray, B: np.ndarray, c_target: np.ndarray,
                 iters: int = 3, scale: float | None = None) -> np.ndarray:
    """Coordinate-descent over one weight row. a0 = initial FP4 row (already scaled)."""
    a = a0.copy().astype(np.float64)
    if scale is None:
        scale = np.max(np.abs(a0)) / 6.0 if np.max(np.abs(a0)) > 0 else 1.0
    s = scale
    best_err = _row_err(a, B, c_target)
    for _ in range(iters):
        changed = False
        for k in range(a.shape[0]):
            orig = a[k]
            local_best = orig
            local_best_err = best_err
            # try every grid value (signed), scaled
            for g in FP4_GRID_SIGNED:
                a[k] = g * s
                err = _row_err(a, B, c_target)
                if err < local_best_err:
                    local_best_err = err
                    local_best = g * s
            a[k] = local_best
            if local_best != orig:
                best_err = local_best_err
                changed = True
        if not changed:
            break
    return a


def backward_quantize(A: np.ndarray, B: np.ndarray,
                      iters: int = 3, progress: bool = False) -> np.ndarray:
    """
    Backward-optimized FP4 quantization of weight matrix A, given activations B.

    Solves, per output row i:  min || a_i @ B - (A@B)_i ||  subject to a_i in FP4 grid.

    Returns A_opt (FP4-encodable: every element is grid_value * scale).
    Caller should also store the per-row scales if per-row scaling is wanted;
    with a single scale this returns a tensor quantizable with one shared scale.
    """
    M = A.shape[0]
    C = A @ B
    A_q_naive, scale = quantize_forward(A)
    A_opt = np.zeros_like(A, dtype=np.float64)
    for i in range(M):
        if progress and (i % max(1, M // 10) == 0):
            print(f"  row {i}/{M}")
        A_opt[i] = optimize_row(A_q_naive[i].copy(), B, C[i], iters=iters, scale=scale)
    return A_opt


def rel_frobenius(C_pred, C_ref) -> float:
    """Relative Frobenius error, percent."""
    return 100.0 * float(np.linalg.norm(C_pred - C_ref) / np.linalg.norm(C_ref))
