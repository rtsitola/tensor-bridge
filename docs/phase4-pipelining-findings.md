# Phase 4 findings — pipelining / LUT decode (honest results)

## What we tried

Phase 4 goal: make the fused FP8 GEMM beat the non-fused (decode + GEMM) path by
hiding the decode behind the tensor-core MMA via pipelining, and by replacing the
costly INT8 decode with a lookup table.

Two concrete attempts, both measured on real GPUs:

1. **LUT decode** (`__constant__ int8_t[256]`): replace `(int8_t)__float2int_rn(__half2float(fp8)*8.0f)`
   (~10 SASS instructions) with a single constant-memory lookup.
2. **Software pipelining** (double-buffering, CUTLASS `cp.async` recipe on Ampere,
   manual 2-stage on Turing) — researched via CUTLASS source + documented here for
   the next implementer.

## Result 1 — the constant LUT does NOT work here (rejected)

Measured on RTX 3070 Ti (Ampere), 512×512×512:

| Path | latency | vs non-fused |
|---|---|---|
| non-fused (decode + GEMM) | 36.5 µs | 1.00× |
| fused (naive decode) | 116.8 µs | 0.31× |
| fused + constant LUT | 423.6 µs | **0.09×** |

The LUT is **3.6× slower than the naive decode** — and wrong (diverges from the
reference). Root cause: the constant cache only *broadcasts* (1 cycle) when all 32
threads read the **same** address. A decode where every element is a different FP8
value produces **random** constant-memory accesses, which serialize (~8–32
cycles/warp). The pipelined ALU decode (which the compiler schedules across the warp)
beats a serialized random constant read.

This is the opposite of the cuda-fp8-ampere benchmark (50.6 TOPS LUT vs 24.5 TOPS
register decode) — their LUT was accessed in a *broadcast* pattern, not random.

## Result 2 — pipelining cannot hide the decode (why fusion loses)

From the CUTLASS recipe research (`mma_pipelined.h`, `mma_multistage.h`,
`memory_sm80.h`):

- **Ampere** pipelines GMEM→SMEM with `cp.async` (3–4 stages), but `cp.async` moves
  *raw bytes* — it does not decode. The FP8→INT8 decode is ALU work that must run
  *after* the copy, on the same warp that issues the MMA. `cp.async` hides the global
  load, not the decode.
- **Turing** has no `cp.async`; double-buffering is `LDG → registers → STS` with
  `__syncthreads`, and it **only overlaps across warps**. With 1 warp/block (our
  tiling), there is nothing to switch to during the load stall — the barrier
  degenerates and the pipeline serializes. The fix (4 warps/block) is a full rewrite
  that still leaves the decode on the critical path.

## The honest conclusion

For **FP8-as-storage on pre-Hopper hardware**, fusion (naive or pipelined) does not
pay:

- The decode is a **memory-bound, embarrassingly-parallel** problem — best run as a
  dedicated kernel with 256 threads/block over the whole array.
- The GEMM is **compute-bound** on tensor cores — best run as a clean kernel without
  decode staging.
- Merging the two puts a serialized decode on the MMA critical path and hurts both.

**The non-fused path (separate decode + GEMM) remains the recommended default.** The
practical win of FP8-as-storage is **memory** (half the VRAM for weights), not compute
speed — and that win does not require fusion.

## What we learned (kept for the next implementer)

- CUTLASS pipeline recipes: `include/cutlass/arch/memory_sm80.h` (cp.async primitives),
  `gemm/threadblock/mma_multistage.h` (Ampere), `mma_pipelined.h` (Turing).
- Constant-memory LUTs only help for broadcast access, never random per-element decode.
- `cp.async` (sm_80+) / no cp.async on Turing → manual double-buffer requires ≥4
  warps/block to overlap.
