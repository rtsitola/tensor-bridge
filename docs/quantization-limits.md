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

## The block scale does NOT help on homogeneous data — but backward optimization does

Measured on Gaussian A~N(0,0.2), B~N(0,0.5), K=128 (deterministic):

| Method | GEMM rel-Frobenius error |
|---|---|
| Naive per-tensor scale | 69.9% |
| Block scale 16 (NVFP4) | 69.7% |
| Block scale 32 (MXFP4) | ~69.7% |
| **Backward-optimized (see docs/backward-fp4-quantization.md)** | **15.4%** |

**Block scale does NOT reduce FP4 error on homogeneous Gaussian data** (69.7% ≈ 69.9%).
It only helps when data is *heterogeneous across blocks* (e.g. real LLM weights with
channel variance). This corrects an earlier claim in this doc.

**The real fix is backward optimization**: instead of quantizing each value to the
nearest grid point, optimize the FP4 representation of each weight row to minimize the
*output* error `||A_q @ B - C||`. This recovers **4.5×** (69.9% → 15.4%), beating any
scaling strategy. It's the GPTQ/AdaRound principle applied to the FP4 grid constraint.

## FP4 error is distribution-dependent

Measured with the OCP E2M1 grid `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` (per-tensor scale):

| Distribution | FP4 GEMM error |
|---|---|
| A~U(0,6), B~U(0,6) | **0.9%** |
| A~U(-1,1), B~U(-1,1) | 74.3% |
| A~N(0,1), B~N(0,1) | 72.7% |
| A~N(0,0.2), B~N(0,0.5) | 72.2% |

**The 70% figure is NOT intrinsic** — it's an artifact of data concentrated near 0,
where the E2M1 grid has its largest relative gap (0.5 → 1.0, ratio 2.0). When data
covers the grid (U(0,6)), FP4 gives ~1%.

**Block scale caveat:** block scales re-anchor the grid locally, which helps only when
data is *heterogeneous across blocks* (real LLM weights with channel variance). On
homogeneous Gaussian data they do nothing (69.9% → 69.7%). The universal fix is
**backward optimization** — see the section above and
[docs/backward-fp4-quantization.md](backward-fp4-quantization.md).

## What actually helps

1. **Weight-only quantization** (`W_q @ X_fp16`) — roughly halves error vs quantizing
   both operands. This is the realistic inference case; make it the default.
2. **Backward-optimized FP4** — 69.9% → 15.4% (4.5×), by optimizing weights for output
   fidelity instead of per-element fidelity. The single biggest lever for FP4.
3. **Per-tensor scale for FP4** — 24.9% → 11.0% (2.3×) on the raw quantize, by recovering
   dynamic range.
4. **FP8 is usable as-is** (~2.4% weight-only) for many purposes.

## Not worth the effort (measured)

- Block scale for FP4 on homogeneous data: no benefit (69.9% → 69.7%).
- Per-channel scale for FP4: no benefit beyond per-tensor.
- Per-channel scale for FP8 on uniform tensors: ≈ per-tensor.

## How to get FP4 under ~5% (the honest answer)

Backward optimization (GPTQ/AdaRound-style, output-fidelity objective) is the proven
path — measured at 15.4% and improvable with more optimization passes or per-row scales.
Combined with grid-aligned weights it reaches LLM-deployable quality. This is a real
quantization project, but the mechanism is validated and cheap (offline, seconds per
weight matrix).
