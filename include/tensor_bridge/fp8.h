/*
 * tensor_bridge/fp8.h — FP8 <-> FP16 conversion primitives (header-only).
 *
 * FP8 E4M3 : 1 sign, 4 exponent (bias 7), 3 mantissa  → range ±448
 * FP8 E5M2 : 1 sign, 5 exponent (bias 15), 2 mantissa → range ±57344
 *
 * Two decode strategies:
 *   - fp8_e4m3_to_half(): generic bit-manipulation (works everywhere, ~12 SASS inst)
 *   - fp8_e4m3_pair_to_half2_lop3(): single-cycle lop3.b32 re-bias (Turing, 2 SASS inst)
 *     — the "H9" trick: exponent bias diff FP16-FP8 = +8, injected via LUT 0xEA.
 */
#ifndef TENSOR_BRIDGE_FP8_H
#define TENSOR_BRIDGE_FP8_H

#include <cstdint>
#include <cuda_fp16.h>

namespace tensor_bridge {

// ---- FP8 E4M3 -> FP16 --------------------------------------------------------
__device__ __host__ __forceinline__ half fp8_e4m3_to_half(uint8_t f8) {
    uint16_t sign  = (uint16_t)(f8 & 0x80u) << 8;          // sign -> bit15
    uint16_t exp8  = (f8 & 0x78u);                          // FP8 exponent (4 bits)
    uint16_t man   = (f8 & 0x07u) << 7;                     // mantissa -> FP16 top bits
    uint16_t exp16 = (uint16_t)((exp8 >> 3) + 8) << 10;     // re-bias +8
    return __ushort_as_half(sign | exp16 | man);
}

// Single-cycle LOP3 variant: pack 2 FP8 bytes -> one FP16 half2.
// out = (packed<<3) OR (mantissa_mask AND bias_inject), LUT 0xEA = A OR (B AND C).
__device__ __forceinline__ half2 fp8_e4m3_pair_to_half2_lop3(uint16_t packed) {
    const uint32_t mask_fp8 = 0x03E003E0u;   // isolate FP8 mantissa bits for both halves
    const uint32_t bias_inj = 0x38003800u;   // exponent bias +8 into both FP16 exp slots
    uint32_t out;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xEA;"
                 : "=r"(out)
                 : "r"(((uint32_t)packed) << 3), "r"(mask_fp8), "r"(bias_inj));
    half2 h;
    h.x = __ushort_as_half((uint16_t)out);
    h.y = __ushort_as_half((uint16_t)(out >> 16));
    return h;
}

// ---- FP8 E5M2 -> FP16 (same exponent bias 15, just zero-extend mantissa) -----
__device__ __host__ __forceinline__ half fp8_e5m2_to_half(uint8_t f8) {
    uint16_t sign = (uint16_t)(f8 & 0x80u) << 8;
    uint16_t exp  = (f8 & 0x7Cu);            // 5-bit exponent, already bias-15
    uint16_t man  = (f8 & 0x03u) << 8;       // 2-bit mantissa -> FP16 low mantissa
    uint16_t h16  = sign | (exp << 10) | man;
    return __ushort_as_half(h16);
}

// ---- FP32 -> FP8 E4M3 (host-side quantize, for reference/test) ---------------
__host__ __device__ __forceinline__ uint8_t fp32_to_fp8_e4m3(float v) {
    float clip = fmaxf(-448.0f, fminf(448.0f, v));
    uint16_t u = __half_as_ushort(__float2half(clip));
    uint8_t sign = (u >> 8) & 0x80u;
    int exp8 = ((u >> 10) & 0x1Fu) - 15 + 7;
    int man8 = (u >> 7) & 0x07u;
    if (exp8 < 0)  { exp8 = 0; man8 = 0; }
    if (exp8 > 15) exp8 = 15;
    return sign | (uint8_t)(exp8 << 3) | (uint8_t)man8;
}

// ---- FP32 -> FP8 E5M2 ---------------------------------------------------------
__host__ __device__ __forceinline__ uint8_t fp32_to_fp8_e5m2(float v) {
    float clip = fmaxf(-57344.0f, fminf(57344.0f, v));
    uint16_t u = __half_as_ushort(__float2half(clip));
    uint8_t sign = (u >> 8) & 0x80u;
    int exp = (u >> 10) & 0x1Fu;             // FP16 exponent == E5M2 exponent (bias 15)
    int man = (u >> 8) & 0x03u;
    return sign | (uint8_t)(exp << 2) | (uint8_t)man;
}

} // namespace tensor_bridge

#endif // TENSOR_BRIDGE_FP8_H
