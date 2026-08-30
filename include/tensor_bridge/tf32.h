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

// TF32 (stored as uint32 with low 13 mantissa bits already dropped) -> FP16
__device__ __host__ __forceinline__ half tf32_to_half(uint32_t tf32) {
    uint32_t sign = (tf32 >> 16) & 0x8000u;
    uint32_t exp  = (tf32 >> 13) & 0x7C00u;   // 8-bit exp -> 5-bit exp field
    uint32_t man  = (tf32 >> 13) & 0x03FFu;   // 10-bit mantissa kept
    uint32_t h16  = sign | exp | man;
    if ((h16 & 0x7C00u) == 0x7C00u) h16 = sign | 0x7C00u; // clamp overflow -> inf
    return __ushort_as_half((uint16_t)h16);
}

// FP32 -> TF32 (round-to-nearest-even, drop 13 mantissa bits)
__host__ __device__ __forceinline__ uint32_t fp32_to_tf32(float v) {
    uint32_t u = f2u(v);
    uint32_t lsb = (u >> 13) & 1u;
    u += 0x00000FFFu + lsb;                    // RNE on 13 dropped bits
    return u & 0xFFFFE000u;
}

} // namespace tensor_bridge

#endif // TENSOR_BRIDGE_TF32_H
