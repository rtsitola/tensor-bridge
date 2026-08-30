/*
 * tensor-bridge/tests/test_conversions.cu — correctness test for all conversion primitives.
 *
 * Verifies FP8 (E4M3/E5M2), BF16, FP4 (E2M1), TF32 decode against known reference
 * values, on device. Run on sm_75 or sm_86 (any).
 */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "tensor_bridge/fp8.h"
#include "tensor_bridge/bf16.h"
#include "tensor_bridge/fp4.h"
#include "tensor_bridge/tf32.h"

#define CHECKVAL(cond, msg) do{ if(!(cond)){ printf("  FAIL: %s\n", msg); fails++; } }while(0)

__global__ void run_tests(int* fails_out) {
    int fails = 0;
    using namespace tensor_bridge;

    // FP8 E4M3 -> FP16 (known values: bias 7, 4 exp bits, 3 mantissa bits)
    // 1.0 = exp 7 (bias 7) man 0 -> 0b0_0111_000 = 0x38;  0.5 = exp 6 -> 0x30;
    // 2.0 = exp 8 -> 0x40;  1.5 = exp 7 man 4 -> 0b0_0111_100 = 0x3C
    CHECKVAL(__half2float(fp8_e4m3_to_half(0x00)) == 0.0f, "e4m3 +0");
    CHECKVAL(__half2float(fp8_e4m3_to_half(0x38)) == 1.0f, "e4m3 +1.0");
    CHECKVAL(__half2float(fp8_e4m3_to_half(0x30)) == 0.5f, "e4m3 +0.5");
    CHECKVAL(__half2float(fp8_e4m3_to_half(0x40)) == 2.0f, "e4m3 +2.0");
    CHECKVAL(__half2float(fp8_e4m3_to_half(0xB8)) == -1.0f, "e4m3 -1.0");
    CHECKVAL(__half2float(fp8_e4m3_to_half(0x3C)) == 1.5f, "e4m3 +1.5");

    // FP8 E5M2 -> FP16
    CHECKVAL(__half2float(fp8_e5m2_to_half(0x3C)) == 1.0f, "e5m2 +1.0");
    CHECKVAL(__half2float(fp8_e5m2_to_half(0x38)) == 0.5f, "e5m2 +0.5");

    // BF16 -> FP16
    CHECKVAL(__half2float(bf16_to_half(0x3F80)) == 1.0f, "bf16 +1.0");
    CHECKVAL(__half2float(bf16_to_half(0x3F00)) == 0.5f, "bf16 +0.5");
    CHECKVAL(__half2float(bf16_to_half(0x4000)) == 2.0f, "bf16 +2.0");
    CHECKVAL(__half2float(bf16_to_half(0xBF80)) == -1.0f, "bf16 -1.0");

    // FP4 E2M1 -> FP16 (LUT: index bits [s e1 e0 m0])
    CHECKVAL(__half2float(fp4_e2m1_to_half(0x2)) == 1.0f, "fp4 +1.0");
    CHECKVAL(__half2float(fp4_e2m1_to_half(0x1)) == 0.5f, "fp4 +0.5");
    CHECKVAL(__half2float(fp4_e2m1_to_half(0x4)) == 2.0f, "fp4 +2.0");
    CHECKVAL(__half2float(fp4_e2m1_to_half(0xA)) == -1.0f, "fp4 -1.0");

    // TF32 -> FP16 (TF32 stored as uint32 with low 13 mantissa bits dropped)
    // 1.0f in TF32 = 0x3F800000 (same as FP32, mantissa already fits 10 bits)
    CHECKVAL(__half2float(tf32_to_half(0x3F800000u)) == 1.0f, "tf32 +1.0");
    CHECKVAL(__half2float(tf32_to_half(0x40000000u)) == 2.0f, "tf32 +2.0");

    // round-trip FP32 -> FP8 E4M3 -> FP16 (quantize accuracy check)
    CHECKVAL(fabsf(__half2float(fp8_e4m3_to_half(fp32_to_fp8_e4m3(1.0f))) - 1.0f) < 1e-6f, "roundtrip 1.0");
    CHECKVAL(fabsf(__half2float(fp8_e4m3_to_half(fp32_to_fp8_e4m3(0.5f))) - 0.5f) < 1e-6f, "roundtrip 0.5");

    fails_out[0] = fails;
}

int main() {
    int *d; cudaMalloc(&d, 4);
    run_tests<<<1,1>>>(d);
    cudaError_t e = cudaDeviceSynchronize();
    int fails; cudaMemcpy(&fails, d, 4, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) { printf("CUDA err %s\n", cudaGetErrorString(e)); return 1; }
    printf("conversion tests: %s (%d fails)\n", fails==0 ? "ALL PASS" : "FAILURES", fails);
    cudaFree(d);
    return fails==0 ? 0 : 1;
}
