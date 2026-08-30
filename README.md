# tensor-bridge

> Run new-generation numeric formats (FP8, FP4, BF16, TF32) on **old tensor cores** that
> never shipped them.

**tensor-bridge** lets a pre-Hopper / pre-Ada GPU execute GEMMs in formats it does not
support natively, by storing tensors in the new format (halving VRAM) and decoding on the
fly into the closest native tensor-core format.

## The matrix (native tensor-core support)

| Format | Volta sm_70 | Turing sm_75 | Ampere sm_86 | Ada sm_89 | Hopper sm_90 | Blackwell sm_120 |
|---|---|---|---|---|---|---|
| FP16 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| BF16 | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| TF32 | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| INT8 / INT4 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| FP8 (E4M3/E5M2) | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| FP4 (NVFP4) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

## What tensor-bridge emulates

| Conversion | Target arch | Mechanism | Status |
|---|---|---|---|
| FP8 E4M3 → FP16 | sm_75 / sm_86 | bit-manipulation re-bias | ✅ tested end-to-end |
| FP8 E4M3 → INT8 | sm_86 | decode + scale → IMMA | ✅ tested |
| BF16 → FP16 | sm_75 / sm_70 | re-bias + clamp | ✅ tested |
| FP4 E2M1 → FP16 | all pre-Blackwell | 16-entry LUT | ✅ tested |
| TF32 → FP16 | sm_75 / sm_70 | re-bias + clamp | ✅ tested |
| FP6 → FP16/INT8 | sm_75 / sm_86 | LUT + re-bias | 🔜 not implemented |
| Fused decode+load GEMM | sm_75 / sm_86 | decode in shared memory | ⚠️ measured SLOWER — see docs/phase3-fusion-findings.md |

> **sm_89 / sm_90 / sm_120 columns are aspirational** — kernels are written and tested for
> sm_75 (Turing) and sm_86 (Ampere) only. Newer archs natively support these formats, so
> tensor-bridge has no value there; they're listed for completeness of the matrix.

## The core idea

A new-gen format's real value is on the tensor cores that *natively* run it (FP8 tensor
cores are 2× INT8 throughput on Hopper/Ada/Blackwell). On older silicon the win is
**memory**, not compute: store weights in 8-bit (or 4-bit) — half the VRAM — and decode to
the native format for the GEMM.

> **Fusion caveat:** the naive fused path (decode inside the GEMM) was measured *slower*
> than decode-then-GEMM on both Turing and Ampere — see
> [docs/phase3-fusion-findings.md](docs/phase3-fusion-findings.md). The recommended path is
> the non-fused one (separate decode kernel + clean tensor-core GEMM). The VRAM saving is
> the win, not speed.

## Layout

```
include/tensor_bridge/   header-only conversion primitives (fp8.h, bf16.h, fp4.h, tf32.h)
src/                     tensor-core GEMM kernels (fp16 path, int8 path) + fused variants
bench/                   benchmark harness (formats × archs × sizes)
tests/                   correctness tests (conversion primitives)
docs/                    nvcc-fill-fragment-bug.md, phase3-fusion-findings.md, phase4-*
```

## Quickstart

```bash
# Turing (Quadro RTX 6000, sm_75) — FP8 → FP16 path
nvcc -O3 -arch=sm_75 -DTARGET_SM75 -Iinclude -o bench_turing bench/bench.cu src/gemm.cu
# Ampere (RTX 3070 Ti, sm_86) — FP8 → INT8 path
nvcc -O3 -arch=sm_86 -DTARGET_SM86 -Iinclude -o bench_ampere bench/bench.cu src/gemm.cu

# Real format test — every format on the right GPU (decode + GEMM, verified vs FP32)
nvcc -O3 -arch=sm_75 -DTARGET_SM75 -Iinclude -o bench_fmts_turing bench/bench_formats.cu src/gemm.cu
nvcc -O3 -arch=sm_86 -DTARGET_SM86 -Iinclude -o bench_fmts_ampere bench/bench_formats.cu src/gemm.cu
```

## Verification & tests

```bash
# Conversion primitives correctness (FP8/BF16/TF32/FP4 decode vs known values)
nvcc -O3 -arch=sm_75 -Iinclude -o test_conv tests/test_conversions.cu
./test_conv   # prints "conversion tests: ALL PASS"
```

Every benchmark result in this README is an average over **50 iterations** of the
decode+GEMM pipeline, and the output is verified against an FP32 reference (relative
Frobenius error reported). See `bench/bench_formats.cu`.

## Measured (256×256×256, decode + GEMM, avg of 50 runs, verified vs FP32)

**Quadro RTX 6000 (Turing sm_75)** — all decoded → FP16:

| Format | GFLOPS | rel-Frobenius error | mantissa |
|---|---|---|---|
| TF32 | 1542 | 0.03% | 10-bit |
| BF16 | 1120 | 0.21% | 7-bit |
| FP8 | 1603 | 3.34% | 3-bit |
| FP4 | 1629 | 37.2% | 1-bit |

**RTX 3070 Ti (Ampere sm_86)** — decoded → FP16:

| Format | GFLOPS | rel-Frobenius error |
|---|---|---|
| FP8 | 1225 | 3.34% |
| FP4 | 1527 | 37.2% |

Error scales exactly with format precision (fewer mantissa bits → more error), as
expected for quantization.

## The compiler bug we documented

Compiling `wmma::fill_fragment()` on **input** fragments (`matrix_a` / `matrix_b`)
silently emits a `BPT.TRAP 0x1` stub instead of HMMA, producing `error 719` at launch.
The workaround is `wmma::load_matrix_sync()` + `col_major` B (see
[docs/nvcc-fill-fragment-bug.md](docs/nvcc-fill-fragment-bug.md)).

## License

MIT
