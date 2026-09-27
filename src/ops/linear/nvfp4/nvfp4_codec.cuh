#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Before native FP4/FP8 conversions (sm_89 for E4M3, sm_100 for E2M1) the cuda_fp4/cuda_fp8
// casts are long generic software sequences that made the NVFP4 GEMV kernels ALU-bound on
// Volta/Turing. Both formats embed exactly in FP16: exponent and mantissa bits placed at the top
// of the FP16 exponent/mantissa fields give the value times 2^-(15 - bias) for normal and
// subnormal encodings alike, so a bit shuffle, the FP16 -> FP32 conversion and an exact
// power-of-two multiply decode them bit for bit.
__device__ __forceinline__ float2 decode_nvfp4_e2m1x2(std::uint8_t storage) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
    const std::uint32_t s    = storage;
    const std::uint32_t bits = ((s & 0x07U) << 9) | ((s & 0x08U) << 12) | ((s & 0x70U) << 21) |
                               ((s & 0x80U) << 24);
    __half2_raw raw;
    raw.x = static_cast<unsigned short>(bits);
    raw.y = static_cast<unsigned short>(bits >> 16);
    const float2 value = __half22float2(__half2(raw));
    return make_float2(value.x * 16384.0F, value.y * 16384.0F);
#else
    __nv_fp4x2_e2m1 value;
    value.__x = storage;
    return static_cast<float2>(value);
#endif
}

__device__ __forceinline__ float decode_nvfp4_e4m3(std::uint8_t storage) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
    // 0x7F/0xFF (NaN) are not produced by the quantizers and are not handled here.
    const std::uint32_t s = storage;
    __half_raw raw;
    raw.x = static_cast<unsigned short>(((s & 0x7FU) << 7) | ((s & 0x80U) << 8));
    return __half2float(__half(raw)) * 256.0F;
#else
    __nv_fp8x2_e4m3 value;
    value.__x = static_cast<std::uint16_t>(storage) | (static_cast<std::uint16_t>(storage) << 8);
    return static_cast<float2>(value).x;
#endif
}

struct alignas(8) Nvfp4QuantizedK16 {
    std::uint32_t codes_lo;
    std::uint32_t codes_hi;
    std::uint8_t scale;
};

static_assert(alignof(Nvfp4QuantizedK16) == 8);

__device__ __forceinline__ void
pack_nvfp4_e2m1x16(const float2 (&values)[8], std::uint32_t& codes_lo, std::uint32_t& codes_hi) {
// cvt.e2m1x2 is native FP4 hardware from sm_80 PTX on; the sm_70 build (Volta, and Turing
// running it) converts in software with the same round-to-nearest-even, saturating semantics.
// Each byte holds the first value of its pair in the low nibble, as the PTX operand order does.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("{\n"
                 ".reg .b8 b0;\n"
                 ".reg .b8 b1;\n"
                 ".reg .b8 b2;\n"
                 ".reg .b8 b3;\n"
                 ".reg .b8 b4;\n"
                 ".reg .b8 b5;\n"
                 ".reg .b8 b6;\n"
                 ".reg .b8 b7;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16;\n"
                 "mov.b32 %0, {b0,b1,b2,b3};\n"
                 "mov.b32 %1, {b4,b5,b6,b7};\n"
                 "}\n"
                 : "=r"(codes_lo), "=r"(codes_hi)
                 : "f"(values[0].x), "f"(values[0].y), "f"(values[1].x), "f"(values[1].y),
                   "f"(values[2].x), "f"(values[2].y), "f"(values[3].x), "f"(values[3].y),
                   "f"(values[4].x), "f"(values[4].y), "f"(values[5].x), "f"(values[5].y),
                   "f"(values[6].x), "f"(values[6].y), "f"(values[7].x), "f"(values[7].y));
#else
    std::uint32_t words[2] = {0, 0};
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const __nv_fp4x2_storage_t byte = __nv_cvt_float2_to_fp4x2(values[pair], __NV_E2M1, cudaRoundNearest);
        words[pair / 4] |= static_cast<std::uint32_t>(byte) << (8 * (pair % 4));
    }
    codes_lo = words[0];
    codes_hi = words[1];
#endif
}

__device__ __forceinline__ Nvfp4QuantizedK16 quantize_nvfp4_k16(const __nv_bfloat16* source,
                                                                float input_scale_divisor) {
    const uint4 packed0                = load_vec<uint4>(source);
    const uint4 packed1                = load_vec<uint4>(source + 8);
    const std::uint32_t represented[8] = {
        packed0.x, packed0.y, packed0.z, packed0.w, packed1.x, packed1.y, packed1.z, packed1.w,
    };

    float2 values[8];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = bf16x2_bits_to_float2(represented[pair]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }

    Nvfp4QuantizedK16 result{};
    const float scale_unencoded = __fdiv_rn(input_scale_divisor * max_abs, 6.0F);
    result.scale                = __nv_cvt_float_to_fp8(scale_unencoded, __NV_SATFINITE, __NV_E4M3);
    if (result.scale == 0) { return result; }

    const float decoded_scale = decode_nvfp4_e4m3(result.scale);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair].x = __fdiv_rn(values[pair].x * input_scale_divisor, decoded_scale);
        values[pair].y = __fdiv_rn(values[pair].y * input_scale_divisor, decoded_scale);
    }
    pack_nvfp4_e2m1x16(values, result.codes_lo, result.codes_hi);
    return result;
}

} // namespace ninfer::ops::detail
