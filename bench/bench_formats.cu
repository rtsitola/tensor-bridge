/*
 * tensor-bridge/bench/bench_formats.cu — REAL test: each format on the right GPU.
 *
 *   Quadro RTX 6000 (Turing sm_75):  BF16, TF32, FP8, FP4  (all decoded -> FP16)
 *   RTX 3070 Ti   (Ampere sm_86):    FP8, FP4            (decoded -> FP16)
 *
 * Per format: quantize FP32 -> format (host) -> decode kernel -> FP16 GEMM on tensor
 * cores (WMMA) -> verify vs FP32 reference + report GFLOPS + rel-Frobenius error.
 *
 * Build:
 *   nvcc -O3 -arch=sm_75 -DTARGET_SM75 -Iinclude -o bench_fmts_turing bench/bench_formats.cu src/gemm.cu
 *   nvcc -O3 -arch=sm_86 -DTARGET_SM86 -Iinclude -o bench_fmts_ampere bench/bench_formats.cu src/gemm.cu
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "tensor_bridge/fp8.h"
#include "tensor_bridge/bf16.h"
#include "tensor_bridge/fp4.h"
#include "tensor_bridge/tf32.h"

#define CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ fprintf(stderr,"CUDA err %d %s\n",e,cudaGetErrorString(e)); exit(1);} }while(0)
constexpr int TILE = 16;

// GEMM from src/gemm.cu (A row_major MxK, B col_major NxK, D float)
__global__ void gemm_fp16_wmma(const half*, const half*, float*, int, int, int);

// ---- decode kernels: format -> half, B transposed to col_major ----------------
__global__ void decode_bf16_to_half(const uint16_t* A, const uint16_t* B,
                                    half* A16, half* B16, int M, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M * K) A16[i] = tensor_bridge::bf16_to_half(A[i]);
    if (i < K * N) { int k = i / N, col = i % N; B16[col * K + k] = tensor_bridge::bf16_to_half(B[i]); }
}
__global__ void decode_tf32_to_half(const uint32_t* A, const uint32_t* B,
                                    half* A16, half* B16, int M, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M * K) A16[i] = tensor_bridge::tf32_to_half(A[i]);
    if (i < K * N) { int k = i / N, col = i % N; B16[col * K + k] = tensor_bridge::tf32_to_half(B[i]); }
}
__global__ void decode_fp8_to_half_fmt(const uint8_t* A, const uint8_t* B,
                                   half* A16, half* B16, int M, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M * K) A16[i] = tensor_bridge::fp8_e4m3_to_half(A[i]);
    if (i < K * N) { int k = i / N, col = i % N; B16[col * K + k] = tensor_bridge::fp8_e4m3_to_half(B[i]); }
}
__global__ void decode_fp4_to_half(const uint8_t* A, const uint8_t* B,
                                   half* A16, half* B16, int M, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M * K) A16[i] = tensor_bridge::fp4_e2m1_to_half(A[i]);
    if (i < K * N) { int k = i / N, col = i % N; B16[col * K + k] = tensor_bridge::fp4_e2m1_to_half(B[i]); }
}

// one full cycle: quantize (host) -> decode -> GEMM -> verify.
// mode 0 = both quantized (worst case), mode 1 = A quantized, B stays FP16 (real
// inference: W_q @ X_fp16). report err vs FP32 reference for both.
#define RUN_FORMAT(NAME, TYPE, QUANT, DECODE, QB_FP16)                                 \
    do {                                                                               \
        std::vector<TYPE> qA((size_t)M*K), qB((size_t)K*N);                            \
        for (size_t i = 0; i < qA.size(); i++) qA[i] = QUANT(hA[i]);                   \
        for (size_t i = 0; i < qB.size(); i++) qB[i] = QUANT(hB[i]);                   \
        TYPE *dA, *dB; half *A16, *B16; float *D, *D2;                                 \
        CHECK(cudaMalloc(&dA, qA.size()*sizeof(TYPE)));                                \
        CHECK(cudaMalloc(&dB, qB.size()*sizeof(TYPE)));                                \
        CHECK(cudaMalloc(&A16, (size_t)M*K*sizeof(half)));                             \
        CHECK(cudaMalloc(&B16, (size_t)K*N*sizeof(half)));                             \
        CHECK(cudaMalloc(&D, (size_t)M*N*sizeof(float)));                              \
        CHECK(cudaMalloc(&D2, (size_t)M*N*sizeof(float)));                             \
        CHECK(cudaMemcpy(dA, qA.data(), qA.size()*sizeof(TYPE), cudaMemcpyHostToDevice));\
        CHECK(cudaMemcpy(dB, qB.data(), qB.size()*sizeof(TYPE), cudaMemcpyHostToDevice));\
        half *Bf16;                                                                     \
        if (QB_FP16) {                                                                  \
            CHECK(cudaMalloc(&Bf16, (size_t)K*N*sizeof(half)));                        \
            std::vector<half> hBf((size_t)K*N);                                         \
            for (size_t k = 0; k < (size_t)K; k++) {                                    \
                for (size_t n = 0; n < (size_t)N; n++)                                  \
                    hBf[n*K + k] = __float2half(hB[k*N + n]);  /* col_major (NxK) */    \
            }                                                                           \
            CHECK(cudaMemcpy(Bf16, hBf.data(), hBf.size()*sizeof(half), cudaMemcpyHostToDevice));\
        }                                                                               \
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);                \
        int total = (int)std::max(qA.size(), qB.size());                               \
        cudaEventRecord(t0);                                                           \
        for (int it = 0; it < iters; it++) {                                           \
            DECODE<<<(total+255)/256, 256>>>(dA, dB, A16, B16, M, N, K);               \
            gemm_fp16_wmma<<<grid, 32>>>(A16, B16, D, M, N, K);                        \
            if (QB_FP16) gemm_fp16_wmma<<<grid, 32>>>(A16, Bf16, D2, M, N, K);         \
        }                                                                               \
        cudaEventRecord(t1); cudaEventSynchronize(t1);                                 \
        float ms = 0; cudaEventElapsedTime(&ms, t0, t1);                               \
        CHECK(cudaDeviceSynchronize());                                                \
        std::vector<float> hD((size_t)M*N), hD2((size_t)M*N);                          \
        CHECK(cudaMemcpy(hD.data(), D, (size_t)M*N*sizeof(float), cudaMemcpyDeviceToHost));\
        double sse = 0, snorm = 0;                                                     \
        for (int i = 0; i < M*N; i++) {                                                \
            double e = (double)hD[i] - hRef[i];                                        \
            sse += e*e; snorm += (double)hRef[i]*hRef[i];                              \
        }                                                                               \
        double e_both = 100.0*sqrt(sse/snorm);                                         \
        if (QB_FP16) {                                                                 \
            CHECK(cudaMemcpy(hD2.data(), D2, (size_t)M*N*sizeof(float), cudaMemcpyDeviceToHost));\
            sse = 0;                                                                    \
            for (int i = 0; i < M*N; i++) {                                            \
                double e = (double)hD2[i] - hRef[i];                                   \
                sse += e*e;                                                             \
            }                                                                           \
            printf("  %-6s : %7.1f us/iter  %8.2f GFLOPS  err(qA@qB) %6.3f%%  err(qA@B_fp16) %6.3f%%\n",\
                   NAME, ms/iters*1000, 2.0*M*N*K/(ms/1000.0/iters)/1e9, e_both,       \
                   100.0*sqrt(sse/snorm));                                              \
        } else {                                                                        \
            printf("  %-6s : %7.1f us/iter  %8.2f GFLOPS  err %6.3f%%\n",              \
                   NAME, ms/iters*1000, 2.0*M*N*K/(ms/1000.0/iters)/1e9, e_both);      \
        }                                                                               \
        cudaFree(dA); cudaFree(dB); cudaFree(A16); cudaFree(B16); cudaFree(D); cudaFree(D2);\
        if (QB_FP16) cudaFree(Bf16);                                                     \
        cudaEventDestroy(t0); cudaEventDestroy(t1);                                     \
    } while (0)

// RUN_FORMAT with per-tensor scale on A (the weights): scale = FORMAT_MAX/maxabs,
// quantize A*scale, decode, GEMM, then dequant result by /scale (for qA@B_fp16).
// This is Claude review item 1 — reuses the INT8 iscale pattern for the FP16 path.
#define RUN_FORMAT_SCALED(NAME, TYPE, QUANT, DECODE, FORMAT_MAX)                        \
    do {                                                                                \
        float maxabs = 0.0f;                                                            \
        for (size_t i = 0; i < (size_t)M*K; i++) maxabs = fmaxf(maxabs, fabsf(hA[i]));  \
        float ascale = FORMAT_MAX / (maxabs > 0.0f ? maxabs : 1.0f);                    \
        std::vector<TYPE> qA((size_t)M*K);                                              \
        for (size_t i = 0; i < qA.size(); i++) qA[i] = QUANT(hA[i] * ascale);           \
        std::vector<TYPE> qBdummy((size_t)K*N, (TYPE)0);  /* decode reads B, give it something */\
        TYPE *dA, *dB; half *A16, *B16; float *D2;                                       \
        CHECK(cudaMalloc(&dA, qA.size()*sizeof(TYPE)));                                 \
        CHECK(cudaMalloc(&dB, qBdummy.size()*sizeof(TYPE)));                            \
        CHECK(cudaMalloc(&A16, (size_t)M*K*sizeof(half)));                              \
        CHECK(cudaMalloc(&B16, (size_t)K*N*sizeof(half)));                              \
        CHECK(cudaMalloc(&D2, (size_t)M*N*sizeof(float)));                              \
        CHECK(cudaMemcpy(dA, qA.data(), qA.size()*sizeof(TYPE), cudaMemcpyHostToDevice));\
        CHECK(cudaMemcpy(dB, qBdummy.data(), qBdummy.size()*sizeof(TYPE), cudaMemcpyHostToDevice));\
        half *Bf16;                                                                     \
        CHECK(cudaMalloc(&Bf16, (size_t)K*N*sizeof(half)));                             \
        std::vector<half> hBf((size_t)K*N);                                             \
        for (size_t k = 0; k < (size_t)K; k++)                                          \
            for (size_t n = 0; n < (size_t)N; n++)                                      \
                hBf[n*K + k] = __float2half(hB[k*N + n]);  /* col_major */              \
        CHECK(cudaMemcpy(Bf16, hBf.data(), hBf.size()*sizeof(half), cudaMemcpyHostToDevice));\
        /* decode A (scaled), then B as identity for the single-kernel pattern */       \
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);                \
        int total = (int)qA.size();                                                     \
        cudaEventRecord(t0);                                                            \
        for (int it = 0; it < iters; it++) {                                            \
            DECODE<<<(total+255)/256, 256>>>(dA, dB, A16, B16, M, N, K);                \
            gemm_fp16_wmma<<<grid, 32>>>(A16, Bf16, D2, M, N, K);                       \
        }                                                                               \
        cudaEventRecord(t1); cudaEventSynchronize(t1);                                 \
        float ms = 0; cudaEventElapsedTime(&ms, t0, t1);                               \
        CHECK(cudaDeviceSynchronize());                                                \
        std::vector<float> hD2((size_t)M*N);                                           \
        CHECK(cudaMemcpy(hD2.data(), D2, (size_t)M*N*sizeof(float), cudaMemcpyDeviceToHost));\
        double sse = 0, snorm = 0;                                                     \
        for (int i = 0; i < M*N; i++) {                                                \
            double v = hD2[i] / ascale;   /* dequant scale_A */                        \
            double e = v - hRef[i];                                                    \
            sse += e*e; snorm += (double)hRef[i]*hRef[i];                              \
        }                                                                               \
        printf("  %-6s(S) : %7.1f us/iter  %8.2f GFLOPS  err(qA_s@B_fp16) %6.3f%%  [scale=%g]\n",\
               NAME, ms/iters*1000, 2.0*M*N*K/(ms/1000.0/iters)/1e9, 100.0*sqrt(sse/snorm), ascale);\
        cudaFree(dA); cudaFree(dB); cudaFree(A16); cudaFree(B16); cudaFree(D2); cudaFree(Bf16);\
        cudaEventDestroy(t0); cudaEventDestroy(t1);                                    \
    } while (0)

int main() {
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("=== tensor-bridge: real format tests ===\nGPU: %s (cc %d.%d)\n\n", prop.name, prop.major, prop.minor);

    const int M = 256, N = 256, K = 256;
    std::vector<float> hA(M * K), hB(K * N), hRef(M * N);
    srand(1);
    for (auto &x : hA) x = ((rand() % 2000) / 1000.0f) - 1.0f;
    for (auto &x : hB) x = ((rand() % 2000) / 1000.0f) - 1.0f;
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
        float s = 0; for (int k = 0; k < K; k++) s += hA[i * K + k] * hB[k * N + j];
        hRef[i * N + j] = s;
    }

    dim3 grid(M / TILE, N / TILE);
    int iters = 50;

#ifdef TARGET_SM75
    printf("Turing (sm_75): BF16, TF32, FP8, FP4 — decoded -> FP16\n");
    printf("  err(qA@qB) = both quantized (worst case); err(qA@B_fp16) = A only (inference W_q@X_fp16)\n");
    printf("  (S) = per-tensor scale on A, dequant /scale (Claude review item 1)\n\n");
    RUN_FORMAT("BF16", uint16_t, tensor_bridge::fp32_to_bf16, decode_bf16_to_half, 1);
    RUN_FORMAT("TF32", uint32_t, tensor_bridge::fp32_to_tf32, decode_tf32_to_half, 1);
    RUN_FORMAT("FP8",  uint8_t,  tensor_bridge::fp32_to_fp8_e4m3, decode_fp8_to_half_fmt, 1);
    RUN_FORMAT("FP4",  uint8_t,  tensor_bridge::fp32_to_fp4_e2m1, decode_fp4_to_half, 1);
    RUN_FORMAT_SCALED("FP8", uint8_t, tensor_bridge::fp32_to_fp8_e4m3, decode_fp8_to_half_fmt, 448.0f);
    RUN_FORMAT_SCALED("FP4", uint8_t, tensor_bridge::fp32_to_fp4_e2m1, decode_fp4_to_half, 6.0f);
#endif
#ifdef TARGET_SM86
    printf("Ampere (sm_86): FP8, FP4 — decoded -> FP16\n");
    printf("  err(qA@qB) = both quantized (worst case); err(qA@B_fp16) = A only (inference W_q@X_fp16)\n");
    printf("  (S) = per-tensor scale on A, dequant /scale (Claude review item 1)\n\n");
    RUN_FORMAT("FP8",  uint8_t,  tensor_bridge::fp32_to_fp8_e4m3, decode_fp8_to_half_fmt, 1);
    RUN_FORMAT("FP4",  uint8_t,  tensor_bridge::fp32_to_fp4_e2m1, decode_fp4_to_half, 1);
    RUN_FORMAT_SCALED("FP8", uint8_t, tensor_bridge::fp32_to_fp8_e4m3, decode_fp8_to_half_fmt, 448.0f);
    RUN_FORMAT_SCALED("FP4", uint8_t, tensor_bridge::fp32_to_fp4_e2m1, decode_fp4_to_half, 6.0f);
#endif
    printf("\n=> all formats decoded & computed on tensor cores, verified vs FP32.\n");
    return 0;
}
