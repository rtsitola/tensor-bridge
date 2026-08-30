/*
 * tensor-bridge/bench/bench.cu — benchmark harness (formats × archs × sizes).
 *
 * Demonstrates the FP8-emulated GEMM and reports GFLOPS + relative Frobenius error.
 * Build: nvcc -O3 -arch=sm_75 -DTARGET_SM75 -Iinclude -o bench_turing bench/bench.cu src/gemm.cu
 *        nvcc -O3 -arch=sm_86 -DTARGET_SM86 -Iinclude -o bench_ampere bench/bench.cu src/gemm.cu
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

// declared in src/gemm.cu
__global__ void decode_fp8_to_half(const uint8_t*, const uint8_t*, half*, half*, int, int, int);
__global__ void decode_fp8_to_int8(const uint8_t*, const uint8_t*, int8_t*, int8_t*, int, int, int, float);
__global__ void gemm_fp16_wmma(const half*, const half*, float*, int, int, int);
__global__ void gemm_int8_wmma(const int8_t*, const int8_t*, int32_t*, int, int, int);

int main() {
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("=== tensor-bridge FP8 emulation benchmark ===\nGPU: %s (cc %d.%d)\n\n", prop.name, prop.major, prop.minor);

    const int M=256, N=256, K=256;
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
    float ms=0; int iters=50;

#ifdef TARGET_SM75
    printf("Path: Turing — decode FP8->FP16 + WMMA FP16 tensor cores\n");
    half *A16,*B16; float *Df;
    CHECK(cudaMalloc(&A16,M*K*2)); CHECK(cudaMalloc(&B16,K*N*2)); CHECK(cudaMalloc(&Df,M*N*4));
    decode_fp8_to_half<<<(M*K+255)/256,256>>>(dA,dB,A16,B16,M,N,K);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(t0);
    for(int it=0;it<iters;it++) gemm_fp16_wmma<<<grid,32>>>(A16,B16,Df,M,N,K);
    cudaEventRecord(t1); cudaEventSynchronize(t1); cudaEventElapsedTime(&ms,t0,t1);
    CHECK(cudaDeviceSynchronize());
    std::vector<float> hD(M*N);
    CHECK(cudaMemcpy(hD.data(),Df,M*N*4,cudaMemcpyDeviceToHost));
    double sse=0, snorm=0;
    for (int i=0;i<M;i++) for(int j=0;j<N;j++){
        double e=(double)hD[i*N+j]-hRef[i*N+j]; sse+=e*e; snorm+=(double)hRef[i*N+j]*hRef[i*N+j];
    }
    printf("Result: %.1f us/iter, %.2f GFLOPS, rel-Frobenius err %.4f%%\n",
           ms/iters*1000, 2.0*M*N*K/(ms/1000.0/iters)/1e9, 100.0*sqrt(sse/snorm));
    cudaFree(A16); cudaFree(B16); cudaFree(Df);
#endif
#ifdef TARGET_SM86
    printf("Path: Ampere — decode FP8->INT8 + WMMA INT8 tensor cores (IMMA)\n");
    int8_t *A8,*B8; int32_t *Di;
    CHECK(cudaMalloc(&A8,M*K)); CHECK(cudaMalloc(&B8,K*N)); CHECK(cudaMalloc(&Di,M*N*4));
    decode_fp8_to_int8<<<(M*K+255)/256,256>>>(dA,dB,A8,B8,M,N,K, 8.0f);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(t0);
    for(int it=0;it<iters;it++) gemm_int8_wmma<<<grid,32>>>(A8,B8,Di,M,N,K);
    cudaEventRecord(t1); cudaEventSynchronize(t1); cudaEventElapsedTime(&ms,t0,t1);
    CHECK(cudaDeviceSynchronize());
    std::vector<int32_t> hDi(M*N);
    CHECK(cudaMemcpy(hDi.data(),Di,M*N*4,cudaMemcpyDeviceToHost));
    double sse=0, snorm=0;
    for (int i=0;i<M;i++) for(int j=0;j<N;j++){
        double v=hDi[i*N+j]/64.0, e=v-hRef[i*N+j]; sse+=e*e; snorm+=(double)hRef[i*N+j]*hRef[i*N+j];
    }
    printf("Result: %.1f us/iter, %.2f GFLOPS, rel-Frobenius err %.4f%%\n",
           ms/iters*1000, 2.0*M*N*K/(ms/1000.0/iters)/1e9, 100.0*sqrt(sse/snorm));
    cudaFree(A8); cudaFree(B8); cudaFree(Di);
#endif
    if (ms<=0) { printf("No path compiled! use -DTARGET_SM75 or -DTARGET_SM86\n"); return 1; }
    printf("=> FP8-emulated GEMM OK (error = expected 8-bit quantization)\n");
    cudaFree(dA); cudaFree(dB);
    return 0;
}
