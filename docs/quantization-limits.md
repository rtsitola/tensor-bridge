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
| FP4 (block scale) | — | see below | — | distribution-dependent |

## FP4 error is distribution-dependent, not a hard 70% limit

Measured with the OCP E2M1 grid `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` (per-tensor scale):

| Distribution | FP4 GEMM error |
|---|---|
| A~U(0,6), B~U(0,6) | **0.9%** |
| A~U(-1,1), B~U(-1,1) | 74.3% |
| A~N(0,1), B~N(0,1) | 72.7% |
| A~N(0,0.2), B~N(0,0.5) | 72.2% |

**The 70% figure is NOT intrinsic** — it's an artifact of data concentrated near 0,
where the E2M1 grid has its largest relative gap (0.5 → 1.0, ratio 2.0). When data
covers the grid (U(0,6)), FP4 gives ~1%. The grid has only 8 magnitudes, but the
*problem* is range-matching, not the grid itself.

**Why NVFP4/MXFP4 use block scales:** they make each block cover its local range, so
every block behaves like the favorable U(0,6) case. Block scale (16 or 32 elements)
genuinely helps when data is concentrated — it re-anchors the grid locally. This
corrects an earlier finding in this doc that claimed scaling cannot help FP4: it can,
specifically via block scales on data concentrated near zero.

**Caveat:** with a per-tensor scale on near-zero data, FP4 is poor (70%). Use block
scales, or ensure the tensor range covers the grid. On grid-covered data, FP4 is
~1% — genuinely usable.

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
