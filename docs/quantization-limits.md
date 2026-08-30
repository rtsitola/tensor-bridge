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

## The block scale helps on heterogeneous data; backward optimization helps everywhere

Measured on Gaussian A~N(0,0.2), B~N(0,0.5), K=128 (deterministic):

| Method | GEMM rel-Frobenius error |
|---|---|
| Naive per-tensor scale | 12.1% |
| Block scale 16 (NVFP4) | ~12% (little change on homogeneous) |
| **Backward-optimized (see docs/backward-fp4-quantization.md)** | **8.5%** |

> **Correction note:** earlier versions of this doc cited 70%/15% (4.5×) from a sign bug
> in the Python validation code. Fixed: naive ~12%, backward ~8.5% (1.4×). The CUDA
> kernel and README benchmark numbers are unaffected.

Block scale helps mostly on **heterogeneous** data (LLM-like weights with channel
variance). On homogeneous Gaussian data it changes little. Backward optimization
(output-fidelity objective, GPTQ/AdaRound-style) helps on both, modestly on homogeneous
(1.4×) and more on heterogeneous weights.

## FP4 error is distribution-dependent

Measured with the OCP E2M1 grid `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` (per-tensor scale):

| Distribution | FP4 GEMM error |
|---|---|
| A~U(0,6), B~U(0,6) | **1.3%** |
| A~U(-1,1), B~U(-1,1) | 11.0% |
| A~N(0,1), B~N(0,1) | 11.8% |
| A~N(0,0.2), B~N(0,0.5) | 12.1% |

**FP4 error depends on whether data covers the grid.** On grid-covered data (U(0,6)) it
is ~1%; on near-zero Gaussian data ~11-12% (per-tensor scale).

**Block scale caveat:** block scales re-anchor the grid locally, which helps when data
is *heterogeneous across blocks* (real LLM weights with channel variance). On
homogeneous Gaussian data they change little. **Backward optimization** helps on both —
see the section above and [docs/backward-fp4-quantization.md](backward-fp4-quantization.md).

## What actually helps

1. **Weight-only quantization** (`W_q @ X_fp16`) — roughly halves error vs quantizing
   both operands. This is the realistic inference case; make it the default.
2. **Backward-optimized FP4** — ~12% → ~8.5% (1.4×), by optimizing weights for output
   fidelity instead of per-element fidelity.
3. **Per-tensor scale for FP4** — 24.9% → 11.0% (2.3×) on the raw quantize, by recovering
   dynamic range.
4. **FP8 is usable as-is** (~2.4% weight-only) for many purposes.

## Not worth the effort (measured)

- Block scale for FP4 on homogeneous data: little benefit.
- Per-channel scale for FP4: no benefit beyond per-tensor.
- Per-channel scale for FP8 on uniform tensors: ≈ per-tensor.

## How to get FP4 under ~5% (the honest answer)

Backward optimization (GPTQ/AdaRound-style, output-fidelity objective) is the proven
path — measured at ~8.5% and improvable with more optimization passes, per-row scales,
and grid-aligned weights. This is a real quantization project, but the mechanism is
validated and cheap (offline, seconds per weight matrix).
