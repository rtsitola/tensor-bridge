/*
 * tensor_bridge/fp4.h — FP4 (E2M1, NVFP4-style) <-> FP16 conversion (header-only).
 *
 * FP4 E2M1 : 1 sign, 2 exponent (bias 1), 1 mantissa → 16 representable values:
 *   ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}  (NVFP4 tops out at ±6, NOT ±8 — max is exp=3,man=1)
 * Blackwell NVFP4 uses this as the core format with per-16-element block scaling.
 */
#ifndef TENSOR_BRIDGE_FP4_H
#define TENSOR_BRIDGE_FP4_H

#include <cstdint>
#include <cuda_fp16.h>

namespace tensor_bridge {

// LUT: FP4 E2M1 (4-bit index) -> FP16 raw bits (unsigned short, static-init safe)
// index bits: [s e1 e0 m0].  value = (1 + m) * 2^(e-1)  (bias 1)
//   idx  magnitude bits   value
//   0    e00 m0  -> 0
//   1    e00 m1  -> 0.5
//   2    e01 m0  -> 1
//   3    e01 m1  -> 1.5
//   4    e10 m0  -> 2
//   5    e10 m1  -> 3
//   6    e11 m0  -> 4
//   7    e11 m1  -> 6   <- 0x4600, NOT 0x45C0 (which is 5.75)
__device__ __constant__ unsigned short kFp4E2M1LUT[16] = {
    0x0000, 0x3800, 0x3C00, 0x3E00, 0x4000, 0x4200, 0x4400, 0x4600,
    0x8000, 0xB800, 0xBC00, 0xBE00, 0xC000, 0xC200, 0xC400, 0xC600,
};

// FP4 E2M1 (4-bit, packed 2-per-byte) -> FP16 via LUT
__device__ __forceinline__ half fp4_e2m1_to_half(uint8_t nibble) {
    return __ushort_as_half(kFp4E2M1LUT[nibble & 0x0Fu]);
}

// Packed: byte holds two FP4 values (hi nibble = first, lo nibble = second)
__device__ __forceinline__ half2 fp4_e2m1_byte_to_half2(uint8_t packed) {
    half2 h;
    h.x = fp4_e2m1_to_half((packed >> 4) & 0x0Fu);   // use the __ushort_as_half path
    h.y = fp4_e2m1_to_half(packed & 0x0Fu);
    return h;
}

// FP32 -> FP4 E2M1 (nearest among the 16 representable values), host-safe
__host__ __device__ __forceinline__ uint8_t fp32_to_fp4_e2m1(float v) {
    static const float vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    int sign = v < 0.0f ? 1 : 0;
    float a = fabsf(v);
    int best = 0; float bd = fabsf(a - vals[0]);
    for (int i = 1; i < 8; i++) { float d = fabsf(a - vals[i]); if (d < bd) { bd = d; best = i; } }
    return (uint8_t)((sign << 3) | best);   // [s e1 e0 m0] matches kFp4E2M1LUT
}

// FP16 -> FP4 E2M1 (round-to-nearest, for reference quantize).
// NVFP4/OCP E2M1 is a TABLE, not a clean formula: index0=0, 1=0.5, 2=1, ...
// 7=6 (no Inf/NaN). So encode by nearest-of-8-magnitudes (reuse fp32_to_fp4_e2m1).
__host__ __device__ __forceinline__ uint8_t half_to_fp4_e2m1(half h) {
    return fp32_to_fp4_e2m1(__half2float(h));
}

} // namespace tensor_bridge

#endif // TENSOR_BRIDGE_FP4_H
