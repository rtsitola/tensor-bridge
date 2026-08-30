# Phase 3 findings — fused decode+load GEMM (honest results)

## What we tried

Fuse the FP8 decode **into** the GEMM kernel (decode into shared memory → `load_matrix_sync`
→ `mma_sync`), to eliminate the separate decode pass and the FP16/INT8 round-trip through
global memory. One kernel instead of two.

Two variants, both correct (verified: identical output vs the non-fused path):

- `gemm_fp8_fused_fp16` (Turing sm_75): FP8 → FP16 in shared → WMMA FP16
- `gemm_fp8_fused_int8` (Ampere sm_86): FP8 → INT8 in shared → WMMA INT8

## Results (512×512×512, 100 iters)

| GPU | non-fused | fused | speedup |
|---|---|---|---|
| Quadro RTX 6000 (Turing) | 72.4 µs | 86.6 µs | **0.84×** (slower) |
| RTX 3070 Ti (Ampere) | 36.4 µs | 112.1 µs | **0.32×** (slower) |

## The honest conclusion: naive fusion is SLOWER

The fused kernel is *not* faster — it is up to 3× slower on Ampere. Two reasons, both
structural:

1. **The decode is serialized with the MMA.** In the fused kernel, the warp must finish
   decoding a whole 16×16 tile (into shared, with `__syncthreads`) before it can issue the
   `mma_sync`. The non-fused version runs decode as a separate, massively-parallel,
   memory-bound kernel (256 threads/block over the whole array) — far better throughput.

2. **The decode is memory-bound, not ALU-bound.** We confirmed this by trying the
   single-cycle `lop3.b32` FP8→FP16 trick (2 SASS inst vs ~12): it made things *worse*,
   not better — because instruction count was never the bottleneck. Loading the `uint8`
   tensor and the shared-memory staging dominate.

## What would actually make fusion win

Software **pipelining** (double-buffering the shared-memory tiles): decode tile `k+1`
while the tensor cores compute tile `k`, so the decode latency is hidden behind the MMA.
This is the technique CUTLASS and cuBLAS use, and it is the real Phase-4 work. It is also
a large effort (async copy via `cp.async` on Ampere+, or manual double-buffer on Turing).

## The practical takeaway for the FP8-on-legacy-GPU goal

The memory *savings* of FP8-as-storage are real (half the VRAM for weights), but the
compute *path* is dominated by the decode staging unless it is pipelined. For a first
release, the **non-fused** path (separate decode + GEMM) is the honest recommendation:
simpler, already correct, and faster than the naive fused version. The fused+pipelined
version is a stretch goal, not the default.
