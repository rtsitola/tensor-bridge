/*
 * tensor-bridge/src/gemm_fused.cu — fused decode+load GEMM kernels.
 *
 * Phase 3: the FP8 decode happens INSIDE the GEMM kernel (into shared memory),
 * eliminating the separate decode pass + the FP16/INT8 round-trip through global
 * memory. One kernel instead of two.
 *
 *   sm_75 (Turing):  load FP8 (uint8) -> decode -> FP16 in shared -> WMMA FP16
 *   sm_86 (Ampere):  load FP8 (uint8) -> decode+scale -> INT8 in shared -> WMMA INT8
 *
 * Pattern (see docs/nvcc-fill-fragment-bug.md):
 *   decode into __shared__ -> wmma::load_matrix_sync() -> mma_sync -> store to global.
 *   NEVER wmma::fill_fragment() on input fragments.
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "tensor_bridge/fp8.h"

using namespace nvcuda;
constexpr int TILE = 16;

// ---- Turing sm_75: FP8 -> FP16 fused GEMM -----------------------------------
__global__ void gemm_fp8_fused_fp16(const uint8_t* A, const uint8_t* B, float* D,
                                    int M, int N, int K) {
    __shared__ __align__(16) half a_s[TILE][TILE];
    __shared__ __align__(16) half b_s[TILE][TILE];
    wmma::fragment<wmma::matrix_a,TILE,TILE,TILE,half,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,TILE,TILE,TILE,half,wmma::col_major> bf;
    wmma::fragment<wmma::accumulator,TILE,TILE,TILE,float> cf;
    wmma::fill_fragment(cf, 0.0f);   // accumulator fill is safe

    int row = blockIdx.x * TILE, col = blockIdx.y * TILE, tid = threadIdx.x;

    for (int k = 0; k < K; k += TILE) {
        // decode FP8 -> FP16 directly into shared (A row_major, B col_major transposed)
        for (int idx = tid; idx < TILE*TILE; idx += blockDim.x) {
            int i = idx / TILE, j = idx % TILE;
            a_s[i][j] = tensor_bridge::fp8_e4m3_to_half(A[(row+i)*K + (k+j)]);
            b_s[j][i] = tensor_bridge::fp8_e4m3_to_half(B[(k+i)*N + (col+j)]);
        }
        __syncthreads();
        wmma::load_matrix_sync(af, &a_s[0][0], TILE);
        wmma::load_matrix_sync(bf, &b_s[0][0], TILE);
        wmma::mma_sync(cf, af, bf, cf);
        __syncthreads();
    }
    wmma::store_matrix_sync(D + row * N + col, cf, N, wmma::mem_row_major);
}

// ---- Ampere sm_86: FP8 -> INT8 fused GEMM -----------------------------------
__global__ void gemm_fp8_fused_int8(const uint8_t* A, const uint8_t* B, int32_t* D,
                                    int M, int N, int K, float scale) {
    __shared__ __align__(16) int8_t a_s[TILE][TILE];
    __shared__ __align__(16) int8_t b_s[TILE][TILE];
    wmma::fragment<wmma::matrix_a,TILE,TILE,TILE,int8_t,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,TILE,TILE,TILE,int8_t,wmma::col_major> bf;
    wmma::fragment<wmma::accumulator,TILE,TILE,TILE,int32_t> cf;
    wmma::fill_fragment(cf, 0);   // accumulator fill is safe

    int row = blockIdx.x * TILE, col = blockIdx.y * TILE, tid = threadIdx.x;

    for (int k = 0; k < K; k += TILE) {
        // decode FP8 -> INT8 (scaled) directly into shared.
        // scale is a per-tensor 127/maxabs so int8 [-128,127] is fully used
        // without wraparound (E4M3 reaches 448; a fixed *8 would overflow).
        for (int idx = tid; idx < TILE*TILE; idx += blockDim.x) {
            int i = idx / TILE, j = idx % TILE;
            half h = tensor_bridge::fp8_e4m3_to_half(A[(row+i)*K + (k+j)]);
            a_s[i][j] = (int8_t)__float2int_rn(__half2float(h) * scale);
            h = tensor_bridge::fp8_e4m3_to_half(B[(k+i)*N + (col+j)]);
            b_s[j][i] = (int8_t)__float2int_rn(__half2float(h) * scale);
        }
        __syncthreads();
        wmma::load_matrix_sync(af, &a_s[0][0], TILE);
        wmma::load_matrix_sync(bf, &b_s[0][0], TILE);
        wmma::mma_sync(cf, af, bf, cf);
        __syncthreads();
    }
    wmma::store_matrix_sync(D + row * N + col, cf, N, wmma::mem_row_major);
}
