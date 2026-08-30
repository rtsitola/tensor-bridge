/*
 * tensor-bridge/bench/bench_fused.cu — Phase 3: fused vs non-fused comparison.
 *
 * Measures END-TO-END pipeline (decode + GEMM) for both approaches:
 *   - non-fused: decode kernel (global round-trip) + gemm kernel  [2 kernels]
 *   - fused:     decode-inside-gemm (shared memory)               [1 kernel]
 * Reports the speedup and verifies identical results.
 *
 * Build:
 *   nvcc -O3 -arch=sm_75 -DTARGET_SM75 -Iinclude -o bench_fused_turing bench/bench_fused.cu src/gemm.cu src/gemm_fused.cu
 *   nvcc -O3 -arch=sm_86 -DTARGET_SM86 -Iinclude -o bench_fused_ampere bench/bench_fused.cu src/gemm.cu src/gemm_fused.cu
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "tensor_bridge/fp8.h"

#define CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ fprintf(stderr,"CUDA err %d %s\n",e,cudaGetErrorString(e)); exit(1);} }while(0)
constexpr int TILE = 16;

// non-fused (src/gemm.cu)
__global__ void decode_fp8_to_half(const uint8_t*, const uint8_t*, half*, half*, int, int, int);
__global__ void decode_fp8_to_int8(const uint8_t*, const uint8_t*, int8_t*, int8_t*, int, int, int);
__global__ void gemm_fp16_wmma(const half*, const half*, float*, int, int, int);
__global__ void gemm_int8_wmma(const int8_t*, const int8_t*, int32_t*, int, int, int);
// fused (src/gemm_fused.cu)
__global__ void gemm_fp8_fused_fp16(const uint8_t*, const uint8_t*, float*, int, int, int);
__global__ void gemm_fp8_fused_int8(const uint8_t*, const uint8_t*, int32_t*, int, int, int);

int main() {
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("=== tensor-bridge Phase 3: fused vs non-fused ===\nGPU: %s (cc %d.%d)\n\n", prop.name, prop.major, prop.minor);

    const int M=512, N=512, K=512;   // bigger than phase-2 to make the round-trip visible
    std::vector<float> hA(M*K), hB(K*N), hRef(M*N);
    srand(1);
    for (auto &x : hA) x = ((rand()%2000)/1000.0f)-1.0f;
    for (auto &x : hB) x = ((rand()%2000)/1000.0f)-1.0f;
    for (int i=0;i<M;i++) for(int j=0;j<N;j++){ float s=0; for(int k=0;k<K;k++) s+=hA[i*K+k]*hB[k*N+j]; hRef[i*N+j]=s; }

    std::vector<uint8_t> qA(M*K), qB(K*N);
    for (size_t i=0;i<M*K;i++) qA[i]=tensor_bridge::fp32_to_fp8_e4m3(hA[i]);
    for (size_t i=0;i<K*N;i++) qB[i]=tensor_bridge::fp32_to_fp8_e4m3(hB[i]);

    uint8_t *dA,*dB;
    CHECK(cudaMalloc(&dA,M*K)); CHECK(cudaMalloc(&dB,K*N));
    CHECK(cudaMemcpy(dA,qA.data(),M*K,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB,qB.data(),K*N,cudaMemcpyHostToDevice));

    dim3 grid(M/TILE, N/TILE);
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    int iters=100;

#ifdef TARGET_SM75
    printf("Path: Turing — FP8 -> FP16 (WMMA)\n\n");
    // --- non-fused ---
    half *A16,*B16; float *Df, *Df2;
    CHECK(cudaMalloc(&A16,M*K*2)); CHECK(cudaMalloc(&B16,K*N*2));
    CHECK(cudaMalloc(&Df,M*N*4)); CHECK(cudaMalloc(&Df2,M*N*4));
    cudaEventRecord(t0);
    for(int it=0;it<iters;it++){
        decode_fp8_to_half<<<(M*K+255)/256,256>>>(dA,dB,A16,B16,M,N,K);
        gemm_fp16_wmma<<<grid,32>>>(A16,B16,Df,M,N,K);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_nf=0; cudaEventElapsedTime(&ms_nf,t0,t1);
    // --- fused ---
    cudaEventRecord(t0);
    for(int it=0;it<iters;it++){
        gemm_fp8_fused_fp16<<<grid,32>>>(dA,dB,Df2,M,N,K);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_f=0; cudaEventElapsedTime(&ms_f,t0,t1);
    CHECK(cudaDeviceSynchronize());

    printf("non-fused: %.1f us/iter\nfused:     %.1f us/iter\nspeedup:   %.2fx\n\n",
           ms_nf/iters*1000, ms_f/iters*1000, ms_nf/ms_f);
    // verify identical
    std::vector<float> h1(M*N), h2(M*N);
    CHECK(cudaMemcpy(h1.data(),Df,M*N*4,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h2.data(),Df2,M*N*4,cudaMemcpyDeviceToHost));
    double maxdiff=0;
    for(int i=0;i<M*N;i++) maxdiff=fmax(maxdiff,fabs((double)h1[i]-h2[i]));
    printf("fused vs non-fused max abs diff: %.6f (expect 0)\n", maxdiff);
    cudaFree(A16); cudaFree(B16); cudaFree(Df); cudaFree(Df2);
#endif
#ifdef TARGET_SM86
    printf("Path: Ampere — FP8 -> INT8 (WMMA/IMMA)\n\n");
    int8_t *A8,*B8; int32_t *Di, *Di2;
    CHECK(cudaMalloc(&A8,M*K)); CHECK(cudaMalloc(&B8,K*N));
    CHECK(cudaMalloc(&Di,M*N*4)); CHECK(cudaMalloc(&Di2,M*N*4));
    cudaEventRecord(t0);
    for(int it=0;it<iters;it++){
        decode_fp8_to_int8<<<(M*K+255)/256,256>>>(dA,dB,A8,B8,M,N,K);
        gemm_int8_wmma<<<grid,32>>>(A8,B8,Di,M,N,K);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_nf=0; cudaEventElapsedTime(&ms_nf,t0,t1);
    cudaEventRecord(t0);
    for(int it=0;it<iters;it++){
        gemm_fp8_fused_int8<<<grid,32>>>(dA,dB,Di2,M,N,K);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_f=0; cudaEventElapsedTime(&ms_f,t0,t1);
    CHECK(cudaDeviceSynchronize());

    printf("non-fused: %.1f us/iter\nfused:     %.1f us/iter\nspeedup:   %.2fx\n\n",
           ms_nf/iters*1000, ms_f/iters*1000, ms_nf/ms_f);
    std::vector<int32_t> h1(M*N), h2(M*N);
    CHECK(cudaMemcpy(h1.data(),Di,M*N*4,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h2.data(),Di2,M*N*4,cudaMemcpyDeviceToHost));
    long long maxdiff=0;
    for(int i=0;i<M*N;i++) maxdiff=std::max(maxdiff, llabs((long long)h1[i]-h2[i]));
    printf("fused vs non-fused max abs diff: %lld (expect 0)\n", maxdiff);
    cudaFree(A8); cudaFree(B8); cudaFree(Di); cudaFree(Di2);
#endif
    cudaFree(dA); cudaFree(dB);
    return 0;
}
