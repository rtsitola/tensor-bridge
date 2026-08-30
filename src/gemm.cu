/*
 * tensor_bridge/src/gemm.cu — tensor-core GEMM kernels, per-architecture dispatch.
 *
 * Two native paths (no FP8 tensor cores required):
 *   sm_75 (Turing):   FP16 WMMA   — decode FP8/BF16/TF32 -> FP16, run fp16 tensor cores
 *   sm_86 (Ampere):   INT8 WMMA   — decode FP8/FP4 -> INT8, run int8 tensor cores (IMMA)
 *
 * IMPORTANT (documented in docs/nvcc-fill-fragment-bug.md):
 *   use wmma::load_matrix_sync() + col_major B. Never wmma::fill_fragment() on input
 *   fragments — it silently compiles to a BPT.TRAP stub (error 719 at launch).
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "tensor_bridge/fp8.h"
#include "tensor_bridge/bf16.h"
#include "tensor_bridge/fp4.h"
#include "tensor_bridge/tf32.h"

using namespace nvcuda;
constexpr int TILE = 16;

// ---- decode kernels (pure ALU, no MMA) --------------------------------------
// A (MxK row_major): decode direct.  B (KxN): decode TRANSPOSED into col_major (NxK).
__global__ void decode_fp8_to_half(const uint8_t* A, const uint8_t* B,
                                   half* A16, half* B16, int M, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M * K) A16[i] = tensor_bridge::fp8_e4m3_to_half(A[i]);
    if (i < K * N) { int k = i / N, col = i % N; B16[col * K + k] = tensor_bridge::fp8_e4m3_to_half(B[i]); }
}
__global__ void decode_fp8_to_int8(const uint8_t* A, const uint8_t* B,
                                   int8_t* A8, int8_t* B8, int M, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M * K) A8[i] = (int8_t)__float2int_rn(__half2float(tensor_bridge::fp8_e4m3_to_half(A[i])) * 8.0f);
    if (i < K * N) { int k = i / N, col = i % N; B8[col * K + k] = (int8_t)__float2int_rn(__half2float(tensor_bridge::fp8_e4m3_to_half(B[i])) * 8.0f); }
}

// ---- GEMM kernels (load_matrix_sync + col_major B, store to global) ---------
__global__ void gemm_fp16_wmma(const half* A, const half* B, float* D, int M, int N, int K) {
    wmma::fragment<wmma::matrix_a,TILE,TILE,TILE,half,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,TILE,TILE,TILE,half,wmma::col_major> bf;
    wmma::fragment<wmma::accumulator,TILE,TILE,TILE,float> cf;
    wmma::fill_fragment(cf, 0.0f);
    int row = blockIdx.x * TILE, col = blockIdx.y * TILE;
    for (int k = 0; k < K; k += TILE) {
        wmma::load_matrix_sync(af, A + row * K + k, K);
        wmma::load_matrix_sync(bf, B + col * K + k, K);
        wmma::mma_sync(cf, af, bf, cf);
    }
    wmma::store_matrix_sync(D + row * N + col, cf, N, wmma::mem_row_major);
}

__global__ void gemm_int8_wmma(const int8_t* A, const int8_t* B, int32_t* D, int M, int N, int K) {
    wmma::fragment<wmma::matrix_a,TILE,TILE,TILE,int8_t,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,TILE,TILE,TILE,int8_t,wmma::col_major> bf;
    wmma::fragment<wmma::accumulator,TILE,TILE,TILE,int32_t> cf;
    wmma::fill_fragment(cf, 0);
    int row = blockIdx.x * TILE, col = blockIdx.y * TILE;
    for (int k = 0; k < K; k += TILE) {
        wmma::load_matrix_sync(af, A + row * K + k, K);
        wmma::load_matrix_sync(bf, B + col * K + k, K);
        wmma::mma_sync(cf, af, bf, cf);
    }
    wmma::store_matrix_sync(D + row * N + col, cf, N, wmma::mem_row_major);
}
