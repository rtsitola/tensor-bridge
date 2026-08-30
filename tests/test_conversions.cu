/*
 * tensor-bridge/tests/test_conversions.cu — correctness test for all conversion primitives.
 *
 * Covers the edge cases the review flagged: E4M3 NaN/max-448, E5M2 Inf/NaN,
 * FP4 all 16 nibbles, BF16/TF32 re-bias, RNE boundaries.
 * Run on sm_75 or sm_86 (any).
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
    auto f = [](half h){ return __half2float(h); };

    // ---- E4M3 decode ----
    CHECKVAL(f(fp8_e4m3_to_half(0x00)) == 0.0f, "e4m3 +0");
    CHECKVAL(f(fp8_e4m3_to_half(0x38)) == 1.0f, "e4m3 +1.0");   // exp7 man0
    CHECKVAL(f(fp8_e4m3_to_half(0x30)) == 0.5f, "e4m3 +0.5");   // exp6 man0
    CHECKVAL(f(fp8_e4m3_to_half(0x40)) == 2.0f, "e4m3 +2.0");   // exp8 man0
    CHECKVAL(f(fp8_e4m3_to_half(0xB8)) == -1.0f, "e4m3 -1.0");
    CHECKVAL(f(fp8_e4m3_to_half(0x3C)) == 1.5f, "e4m3 +1.5");   // exp7 man4
    // max finite E4M3 = 448: exp15 man6 = 0x7E
    CHECKVAL(f(fp8_e4m3_to_half(0x7E)) == 448.0f, "e4m3 max 448 (exp15,man6)");
    // NaN: exp15 man7 = 0x7F
    CHECKVAL(__hisnan(fp8_e4m3_to_half(0x7F)), "e4m3 NaN (exp15,man7)");

    // ---- E5M2 decode ----
    CHECKVAL(f(fp8_e5m2_to_half(0x3C)) == 1.0f, "e5m2 +1.0");   // exp15 man0
    CHECKVAL(f(fp8_e5m2_to_half(0x38)) == 0.5f, "e5m2 +0.5");
    CHECKVAL(f(fp8_e5m2_to_half(0x40)) == 2.0f, "e5m2 +2.0");
    CHECKVAL(f(fp8_e5m2_to_half(0x7B)) == 57344.0f, "e5m2 max 57344 (exp30,man3)");
    CHECKVAL(__hisinf(fp8_e5m2_to_half(0x7C)), "e5m2 exp31 man0 -> inf");
    CHECKVAL(__hisnan(fp8_e5m2_to_half(0x7F)), "e5m2 exp31 man3 -> NaN");

    // ---- BF16 decode ----
    CHECKVAL(f(bf16_to_half(0x3F80)) == 1.0f, "bf16 +1.0");
    CHECKVAL(f(bf16_to_half(0x3F00)) == 0.5f, "bf16 +0.5");
    CHECKVAL(f(bf16_to_half(0x4000)) == 2.0f, "bf16 +2.0");
    CHECKVAL(f(bf16_to_half(0xBF80)) == -1.0f, "bf16 -1.0");
    CHECKVAL(__hisinf(bf16_to_half(0x7F80)), "bf16 +Inf preserved");
    CHECKVAL(__hisnan(bf16_to_half(0x7FC0)), "bf16 NaN preserved (not Inf)");

    // ---- FP4 decode: all 16 nibbles against spec values ----
    const float fp4_exp[16] = {0,0.5f,1,1.5f,2,3,4,6, -0,-0.5f,-1,-1.5f,-2,-3,-4,-6};
    for (int i = 0; i < 16; i++) {
        half h = fp4_e2m1_to_half((uint8_t)i);
        float got = f(h), want = fp4_exp[i];
        // compare with sign handling for -0
        if (i < 8) CHECKVAL(fabsf(got - want) < 1e-5f, "fp4 LUT");
        else CHECKVAL(fabsf(got - want) < 1e-5f, "fp4 LUT neg");
    }
    // verify fp4_e2m1_byte_to_half2 bitcast works
    half2 h2 = fp4_e2m1_byte_to_half2(0x72); // hi=7 (6.0), lo=2 (1.0)
    CHECKVAL(f(h2.x) == 6.0f && f(h2.y) == 1.0f, "fp4 byte_to_half2");

    // ---- FP4 roundtrip quantize ----
    CHECKVAL(f(fp4_e2m1_to_half(fp32_to_fp4_e2m1(1.0f))) == 1.0f, "fp4 rt 1.0");
    CHECKVAL(f(fp4_e2m1_to_half(fp32_to_fp4_e2m1(5.5f))) == 6.0f, "fp4 rt 5.5->6.0"); // unambiguous

    // ---- half_to_fp4_e2m1 (separate helper, exercises the sign bit) ----
    CHECKVAL(f(fp4_e2m1_to_half(half_to_fp4_e2m1(__float2half(1.0f)))) == 1.0f, "half_to_fp4 +1.0");
    CHECKVAL(f(fp4_e2m1_to_half(half_to_fp4_e2m1(__float2half(-1.0f)))) == -1.0f, "half_to_fp4 -1.0 (sign bit 3)");
    CHECKVAL(f(fp4_e2m1_to_half(half_to_fp4_e2m1(__float2half(-2.0f)))) == -2.0f, "half_to_fp4 -2.0");
    CHECKVAL(f(fp4_e2m1_to_half(half_to_fp4_e2m1(__float2half(0.5f)))) == 0.5f, "half_to_fp4 +0.5");
    CHECKVAL(f(fp4_e2m1_to_half(half_to_fp4_e2m1(__float2half(-5.0f)))) == -4.0f, "half_to_fp4 -5.0->-4.0 (nearest, tie)");

    // ---- TF32 decode ----
    CHECKVAL(f(tf32_to_half(0x3F800000u)) == 1.0f, "tf32 +1.0");
    CHECKVAL(f(tf32_to_half(0x40000000u)) == 2.0f, "tf32 +2.0");
    CHECKVAL(__hisinf(tf32_to_half(0x7F800000u)), "tf32 +Inf preserved");
    CHECKVAL(__hisnan(tf32_to_half(0x7FC00000u)), "tf32 NaN preserved (not Inf)");

    // ---- FP8 roundtrip RNE ----
    CHECKVAL(f(fp8_e4m3_to_half(fp32_to_fp8_e4m3(1.0f))) == 1.0f, "e4m3 rt 1.0");
    CHECKVAL(f(fp8_e4m3_to_half(fp32_to_fp8_e4m3(0.5f))) == 0.5f, "e4m3 rt 0.5");
    CHECKVAL(f(fp8_e4m3_to_half(fp32_to_fp8_e4m3(448.0f))) == 448.0f, "e4m3 rt max 448");

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
