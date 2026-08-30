/*
 * tensor_bridge/bf16.h — BF16 <-> FP16 conversion (header-only).
 *
 * BF16 : 1 sign, 8 exponent (bias 127), 7 mantissa  → range ±3.4e38
 * FP16 : 1 sign, 5 exponent (bias 15), 10 mantissa  → range ±65504
 *
 * BF16 -> FP16 is NOT lossless: FP16 has a smaller exponent range but MORE mantissa
 * precision. Values with |x| > 65504 clamp to ±inf; the mantissa top 7 bits map cleanly.
 * This is the standard path to run BF16 models on Turing/Volta (no native BF16).
 */
#ifndef TENSOR_BRIDGE_BF16_H
#define TENSOR_BRIDGE_BF16_H

#include <cstdint>
#include <cuda_fp16.h>
#include "tensor_bridge/common.h"

namespace tensor_bridge {

// ---- BF16 -> FP16 (lossy: clamp out-of-range, keep top 7 mantissa bits) ------
// BF16 bias 127, FP16 bias 15 -> exponent offset = -112. Denormals -> +-0 (approx).
__device__ __host__ __forceinline__ half bf16_to_half(uint16_t bf) {
    uint16_t sign = (bf & 0x8000u);
    int e = ((bf >> 7) & 0xFFu) - 112;             // re-bias 127 -> 15
    uint16_t man = (bf & 0x007Fu) << 3;            // 7-bit mantissa -> FP16 top 7
    if (e <= 0) return __ushort_as_half(sign);     // denormal -> +-0 (approx)
    if (e >= 31) return __ushort_as_half(sign | 0x7C00u); // overflow -> +-inf
    return __ushort_as_half(sign | (uint16_t)(e << 10) | man);
}

// ---- FP32 -> BF16 ------------------------------------------------------------
__host__ __device__ __forceinline__ uint16_t fp32_to_bf16(float v) {
    uint32_t u = f2u(v);
    uint16_t lsb = (u >> 16) & 1u;
    uint16_t round_bit = 0x7FFFu + lsb;
    u += round_bit;
    return (uint16_t)(u >> 16);
}

// ---- FP16 -> BF16 (correct: expand to FP32, then RNE-drop 16 mantissa bits) ----
// A true BF16 has an 8-bit exponent (bias 127), so we must route through FP32 —
// NOT just drop low FP16 mantissa bits (that would keep the 5-bit FP16 exponent).
__host__ __device__ __forceinline__ uint16_t half_to_bf16(half h) {
    return fp32_to_bf16(__half2float(h));
}

} // namespace tensor_bridge

#endif // TENSOR_BRIDGE_BF16_H
