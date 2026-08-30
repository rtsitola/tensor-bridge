/*
 * tensor_bridge/tf32.h — TF32 <-> FP16 conversion (header-only).
 *
 * TF32 : 1 sign, 8 exponent (bias 127), 10 mantissa  (FP32 with reduced mantissa)
 * FP16 : 1 sign, 5 exponent (bias 15), 10 mantissa
 *
 * TF32 -> FP16 is lossy on RANGE (FP16 max normal ±65504 vs TF32 ±3.4e38) but keeps
 * the full 10-bit mantissa. For the magnitudes typical in GEMM activations this is
 * fine; out-of-range values clamp to ±inf. Used to run TF32 models on Volta/Turing.
 */
#ifndef TENSOR_BRIDGE_TF32_H
#define TENSOR_BRIDGE_TF32_H

#include <cstdint>
#include <cuda_fp16.h>
#include "tensor_bridge/common.h"

namespace tensor_bridge {

// TF32 (uint32, low 13 mantissa bits already dropped) -> FP16
// TF32 = FP32 with 10-bit mantissa: bias 127 -> FP16 bias 15 = -112. Denormal -> +-0.
__device__ __host__ __forceinline__ half tf32_to_half(uint32_t tf32) {
    uint16_t sign = (uint16_t)((tf32 >> 16) & 0x8000u);
    int e = ((tf32 >> 23) & 0xFFu) - 112;           // re-bias 127 -> 15
    uint16_t man = (uint16_t)((tf32 >> 13) & 0x03FFu); // 10-bit mantissa
    if (e <= 0) return __ushort_as_half(sign);      // denormal -> +-0 (approx)
    if (e >= 31) {
        // preserve NaN vs Inf (exp all-ones in TF32 = 0xFF, mantissa is 23 bits)
        if (((tf32 >> 23) & 0xFFu) == 0xFFu && (tf32 & 0x007FFFFFu)) // TF32 NaN
            return __ushort_as_half(sign | 0x7E00u);                // FP16 NaN (quiet)
        return __ushort_as_half(sign | 0x7C00u);    // overflow -> +-Inf
    }
    return __ushort_as_half(sign | (uint16_t)(e << 10) | man);
}

// FP32 -> TF32 (round-to-nearest-even, drop 13 mantissa bits)
__host__ __device__ __forceinline__ uint32_t fp32_to_tf32(float v) {
    uint32_t u = f2u(v);
    // preserve NaN: rounding could overflow mantissa into exponent (NaN -> Inf)
    if ((u & 0x7F800000u) == 0x7F800000u) return u & 0xFFFFE000u; // Inf/NaN keep sign
    uint32_t lsb = (u >> 13) & 1u;
    u += 0x00000FFFu + lsb;                    // RNE on 13 dropped bits
    return u & 0xFFFFE000u;
}

} // namespace tensor_bridge

#endif // TENSOR_BRIDGE_TF32_H
