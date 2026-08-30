# nvcc silently compiles `wmma::fill_fragment` (input fragments) to a `BPT.TRAP` stub

**Status:** confirmed, workaround documented. Affects nvcc 12.4 and 13.3 (and likely
earlier). Hardware is not at fault — `cuBLAS` FP16 runs fine on the same GPUs.

## Symptom

A kernel that builds `wmma::fragment` inputs with `wmma::fill_fragment()` compiles
**without error**, but the entire kernel body is emitted as a trap stub:

```sass
        /*0000*/  MOV R1, c[0x0][0x28] ;
        /*0010*/  BPT.TRAP 0x1 ;        ;  <- whole body is this
        /*0020*/  BRA 0x20 ;             ;  <- infinite loop
        /*0030*/  NOP ; ...              ;  <- empty
```

The cubin header carries `EF_CUDA_VIRTUAL_SM(...)`, and at launch the kernel fails with
`CUDA error 719 (unspecified launch failure)`.

## Minimal repro

```cuda
__global__ void mm(const half* A, const half* B, float* D){
  wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af;
  wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bf;
  wmma::fragment<wmma::accumulator,16,16,16,float> cf;
  wmma::fill_fragment(af, __float2half(1.0f));   // <-- triggers the trap stub
  wmma::fill_fragment(bf, __float2half(1.0f));   // <--
  wmma::fill_fragment(cf, 0.0f);                 // fill on accumulator is FINE
  wmma::mma_sync(cf, af, bf, cf);
  wmma::store_matrix_sync(D, cf, 16, wmma::mem_row_major);
}
// mm<<<1,32>>>(...) -> error 719
```

`cuobjdump -sass` on the binary shows `MOV + BPT.TRAP 0x1 + BRA` and nothing else.

## Root cause

`ptxas` fails to lower the WMMA operand setup that `fill_fragment` on **input**
fragments produces, and instead of erroring it emits a `BPT.TRAP` stub. The `EF_CUDA_VIRTUAL_SM`
marker shows the cubin is a placeholder, not real SASS. The valid PTX is still embedded
in the fatbin, but the driver loads the (trap) cubin because it matches the target SM.

Related public reports of the same mechanism (ptxas failure → trap stub):

- Triton issue #9933 — `ptxas` internal error C7907 → `trap;` stub:
  https://github.com/triton-lang/triton/issues/9933
- NVIDIA forums — "PTX and SASS codes corresponding to device code are empty":
  https://forums.developer.nvidia.com/t/285211
- NVIDIA forums — "mma not lowered to tensor core SASS":
  https://forums.developer.nvidia.com/t/208808
- NVIDIA forums — WMMA is now a "compatibility / fallback" interface, prefer `mma.sync`:
  https://forums.developer.nvidia.com/t/361063

## Workarounds (in order of preference)

1. **`wmma::load_matrix_sync()` from memory + `col_major` B** (what this repo uses).
   `fill_fragment` is only safe on the **accumulator** fragment.
2. **`CUDA_FORCE_PTX_JIT=1`** — force the driver to JIT the embedded PTX instead of
   loading the trap cubin (no recompile needed).
3. **Inline `mma.sync` PTX** — bypass `wmma.h` entirely:
   ```cuda
   asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
   ```
4. Explicit `-gencode arch=compute_XX,code=sm_XX` (native cubin, no JIT path).

## Verification (how to confirm it's the trap, not your code)

```bash
cuobjdump -sass app | grep -E "BPT.TRAP|HMMA"
# BPT.TRAP present + no HMMA -> trap stub. HMMA present -> real code.
```
