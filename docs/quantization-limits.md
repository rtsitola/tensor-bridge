# Quantization limits — what scaling can and cannot do

Consolidated findings from measuring the FP8/FP4 decode+GEMM on Turing (sm_75) and
Ampere (sm_86), cross-checked by Grok and Claude reviews.

## The bottom line

| Format | err(qA@qB) | err(qA@B_fp16) | + per-tensor scale | What scaling buys |
|---|---|---|---|---|
| TF32 | 0.03% | 0.03% | — | nothing needed |
| BF16 | 0.21% | 0.15% | — | nothing needed |
| FP8 | 3.34% | 2.37% | 2.54% | little (scale≈448 near-identity on [-1,1]) |
| FP4 | 37.2% | 24.9% | 11.0% | 2.3× — recovers dynamic range |
| FP4 (block scale) | — | ~71% | — | **nothing** (see below) |

## FP4 has a hard 4-bit precision limit — scaling can't fix it

Measured on Gaussian matrices: FP4 error stays ~70% regardless of scale strategy —
per-tensor 71.3%, per-row 71.4%, block-32 71.2%, block-16 71.1%. A "perfect" per-element
scale also gives ~72%.

**Why:** the E2M1 grid has only 8 magnitudes `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` with a
relative ratio of **2.0 between 0.5 and 1.0** (50% spacing). Scaling shifts the grid but
does not densify it. The error is the intrinsic 4-bit mantissa resolution, not a range
problem.

**Implication:** NVFP4 / OCP MXFP4 is only usable for **weights that are pre-quantized to
align with the grid** (AWQ/GPTQ calibration), never for arbitrary/continuous data. Don't
ship "FP4 precision" numbers as if block scaling will fix them — it won't.

## Per-channel scale helps FP8 a little, only on heterogeneous channels

On uniform `[-1,1]`, per-row ≈ per-tensor (rows have similar max). On realistic
heterogeneous channels (1/3 of rows 10× larger), FP8 per-row gives 2.63% → 2.55% —
a real but small gain. The format error is dominated by mantissa precision, not range.

## What actually helps

1. **Weight-only quantization** (`W_q @ X_fp16`) — roughly halves error vs quantizing
   both operands. This is the realistic inference case; make it the default.
2. **Per-tensor scale for FP4** — 24.9% → 11.0% (2.3×), by recovering dynamic range
   (naked E2M1 on [-1,1] collapses to {0, ±0.5, ±1}).
3. **FP8 is usable as-is** (~2.4% weight-only) for many purposes.

## Not worth the effort (measured)

- Per-channel scale for FP4: no benefit (precision limit dominates).
- Block scale for FP4 on arbitrary data: no benefit (same reason).
- Per-channel scale for FP8 on uniform tensors: ≈ per-tensor.

## How to get FP4 under ~5% (the honest answer)

Only via MXFP4/NVFP4 **with block scales AND grid-aligned weights** (AWQ/GPTQ-style
calibration), on top of a real inference pipeline. That's a separate, substantial project
— not a tweak to the GEMM.
