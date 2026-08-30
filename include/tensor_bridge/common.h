/*
 * tensor_bridge/common.h — shared portable helpers.
 */
#ifndef TENSOR_BRIDGE_COMMON_H
#define TENSOR_BRIDGE_COMMON_H

#include <cstdint>
#include <cstring>

namespace tensor_bridge {

// portable float <-> uint32 bit-cast (__float_as_uint is device-only)
__host__ __device__ __forceinline__ uint32_t f2u(float v) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(v);
#else
    uint32_t u; memcpy(&u, &v, sizeof(u)); return u;
#endif
}

__host__ __device__ __forceinline__ float u2f(uint32_t u) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(u);
#else
    float v; memcpy(&v, &u, sizeof(v)); return v;
#endif
}

} // namespace tensor_bridge

#endif // TENSOR_BRIDGE_COMMON_H
