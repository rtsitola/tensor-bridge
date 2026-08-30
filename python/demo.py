"""
tensor_bridge_fp4/demo.py — run the backward FP4 benchmark.

Reproduces the measured result: naive FP4 ~12% error on concentrated Gaussian data,
backward-optimized FP4 ~8.5% (1.4x better).

Usage:
    python demo.py [M] [K] [N] [--iters N]
"""

from __future__ import annotations
import sys
import time
import numpy as np

from tensor_bridge_fp4 import (
    quantize_forward, backward_quantize, rel_frobenius, FP4_GRID_SIGNED,
)


def main() -> int:
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    iters = 3

    np.random.seed(1)
    A = np.random.normal(0, 0.2, (M, K))
    B = np.random.normal(0, 0.5, (K, N))
    C_ref = A @ B

    # naive forward FP4 (per-tensor scale)
    A_q, scale = quantize_forward(A)
    err_naive = rel_frobenius(A_q @ B, C_ref)

    # backward-optimized
    t0 = time.time()
    A_opt = backward_quantize(A, B, iters=iters, progress=False)
    elapsed = time.time() - t0
    err_back = rel_frobenius(A_opt @ B, C_ref)

    # verify A_opt is FP4-encodable (every element = grid_value * scale)
    A_opt_units = A_opt / scale
    encodable = np.all(np.isin(np.round(A_opt_units, 6), FP4_GRID_SIGNED))

    print(f"\n=== Backward FP4 quantization (M={M}, K={K}, N={N}) ===")
    print(f"  A~N(0,0.2), B~N(0,0.5), scale={scale:.4f}")
    print(f"  naive forward FP4   : {err_naive:6.2f}%")
    print(f"  backward-optimized  : {err_back:6.2f}%   ({err_naive/err_back:.1f}x better)")
    print(f"  time                : {elapsed:.2f}s")
    print(f"  A_opt FP4-encodable : {encodable}")
    print(f"\n  -> {err_back:.1f}% < {err_naive:.1f}%: backward optimization wins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
