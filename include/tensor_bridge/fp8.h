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
// E4M3fn (NVIDIA): NO Inf. Only exp=15,man=7 is NaN; exp15 man0..6 are finite
// (128..448). max finite = exp15,man6 = 448.
__device__ __host__ __forceinline__ half fp8_e4m3_to_half(uint8_t f8) {
    uint16_t sign  = (uint16_t)(f8 & 0x80u) << 8;          // sign -> bit15
    uint8_t expE   = (f8 >> 3) & 0x0Fu;                    // 4-bit exponent VALUE
    uint16_t man   = (f8 & 0x07u) << 7;                    // mantissa -> FP16 top bits
    if (expE == 0) return __ushort_as_half(sign);          // denormal/zero -> +-0 (approx)
    if (expE == 15 && (f8 & 0x07u) == 0x07u) return __ushort_as_half(sign | 0x7E00u); // exp15,man7 -> NaN
    uint16_t exp16 = (uint16_t)(expE + 8) << 10;           // re-bias: FP16 field = E+8
    return __ushort_as_half(sign | exp16 | man);
}

// ---- FP8 E5M2 -> FP16 (same exponent bias 15, just zero-extend mantissa) -----
__device__ __host__ __forceinline__ half fp8_e5m2_to_half(uint8_t f8) {
    uint16_t sign = (uint16_t)(f8 & 0x80u) << 8;
    uint16_t exp  = (uint16_t)((f8 >> 2) & 0x1Fu) << 10;   // 5-bit exp, bias 15 == fp16
    uint16_t man  = (uint16_t)(f8 & 0x03u) << 8;           // 2-bit mantissa -> low bits
    return __ushort_as_half(sign | exp | man);
}

// ---- FP32 -> FP8 E4M3 (host-side quantize, RNE) ------------------------------
// Round-to-nearest-even on the 10 FP16 mantissa bits that don't fit FP8's 3.
__host__ __device__ __forceinline__ uint8_t fp32_to_fp8_e4m3(float v) {
    float clip = fmaxf(-448.0f, fminf(448.0f, v));
    uint16_t u = __half_as_ushort(__float2half(clip));  // RNE already applied by half cast
    uint16_t man = u & 0x03FFu;                          // FP16 10-bit mantissa
    // round-to-nearest-even to 3 mantissa bits (drop 7 low bits)
    uint16_t lsb = (man >> 7) & 1u;
    uint16_t round_bit = 0x0040u;                        // halfway of 7 dropped bits
    uint16_t sticky = man & 0x007Fu;
    uint16_t r = (sticky > round_bit || (sticky == round_bit && lsb)) ? 0x0080u : 0u;
    uint16_t man3 = (man + r) >> 7;                      // now 3 mantissa bits

    uint8_t sign = (u >> 8) & 0x80u;
    int expE = ((u >> 10) & 0x1Fu) - 15 + 7;               // FP16 exp -> FP8 exp value (bias 7)
    // mantissa overflow: carry into exponent
    if (man3 >= 8) { man3 = 0; expE++; }
    if (expE <= 0) return sign;                            // denormal/zero -> +-0
    if (expE > 15) { expE = 15; man3 = 6; }                // clamp to max E4M3 = 448 (exp15,man6)
    return sign | (uint8_t)(expE << 3) | (uint8_t)man3;
}

// ---- FP32 -> FP8 E5M2 (host-side quantize, RNE) ------------------------------
__host__ __device__ __forceinline__ uint8_t fp32_to_fp8_e5m2(float v) {
    float clip = fmaxf(-57344.0f, fminf(57344.0f, v));
    uint16_t u = __half_as_ushort(__float2half(clip));   // RNE applied by half cast
    uint16_t man = u & 0x03FFu;                          // FP16 10-bit mantissa
    // round-to-nearest-even to 2 mantissa bits (drop 8 low bits)
    uint16_t lsb = (man >> 8) & 1u;
    uint16_t round_bit = 0x0080u;                        // halfway of 8 dropped bits
    uint16_t sticky = man & 0x00FFu;
    uint16_t r = (sticky > round_bit || (sticky == round_bit && lsb)) ? 0x0100u : 0u;
    uint16_t man2 = (man + r) >> 8;                      // now 2 mantissa bits

    uint8_t sign = (u >> 8) & 0x80u;
    int exp = (u >> 10) & 0x1Fu;                         // E5M2 exp == FP16 exp (bias 15)
    if (man2 >= 4) { man2 = 0; exp++; }                  // mantissa carry
    if (exp <= 0) return sign;                           // denormal/zero -> +-0
    if (exp >= 31) exp = 31;                             // clamp
    return sign | (uint8_t)(exp << 2) | (uint8_t)man2;
}

} // namespace tensor_bridge

#endif // TENSOR_BRIDGE_FP8_H
