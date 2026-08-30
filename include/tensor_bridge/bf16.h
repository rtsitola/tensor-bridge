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
__device__ __host__ __forceinline__ half bf16_to_half(uint16_t bf) {
    uint16_t sign = (bf & 0x8000u);          // sign -> bit15 (same position)
    uint16_t exp  = (bf & 0x7F80u) >> 3;     // 8-bit exp -> 5-bit exp (bias 127 -> 15)
    uint16_t man  = (bf & 0x007Fu) << 3;     // 7-bit mantissa -> FP16 top 7 of 10 bits
    uint16_t h16  = sign | exp | man;
    // clamp: FP16 max normal exponent field is 0x7C00 (exp 31 = inf/nan)
    if ((h16 & 0x7C00u) == 0x7C00u) {
        // overflow -> +-inf, keep sign
        h16 = sign | 0x7C00u;
    }
    return __ushort_as_half(h16);
}

// ---- FP16 -> BF16 (round-to-nearest-even on the dropped mantissa bits) ------
__device__ __host__ __forceinline__ uint16_t half_to_bf16(half h) {
    uint16_t u = __half_as_ushort(h);
    uint16_t man = u & 0x03FFu;              // FP16 10-bit mantissa
    uint16_t lsb = (man >> 3) & 1u;          // bit that becomes the LSB after truncation
    uint16_t round_bit = 0x0004u;            // halfway point (0.5 * 2^3)
    uint16_t sticky = man & 0x0007u;         // the 3 bits being dropped
    uint16_t r = 0;
    if (sticky > round_bit || (sticky == round_bit && lsb)) r = 0x0008u; // RNE
    return (u + r) & 0xFFE0u;                // drop low 3 mantissa bits
}

// ---- FP32 -> BF16 ------------------------------------------------------------
__host__ __device__ __forceinline__ uint16_t fp32_to_bf16(float v) {
    uint32_t u = f2u(v);
    uint16_t lsb = (u >> 16) & 1u;
    uint16_t round_bit = 0x7FFFu + lsb;
    u += round_bit;
    return (uint16_t)(u >> 16);
}

} // namespace tensor_bridge

#endif // TENSOR_BRIDGE_BF16_H
