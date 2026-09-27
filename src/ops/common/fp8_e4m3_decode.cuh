#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops {

// Two E4M3FN codes (low byte first) as an exact __half2. Before sm_89 the cuda_fp8 cast is a
// long software sequence; there the code bits placed at FP16 bits 13..7 (sign at 15) are the
// value times 2^-8 for normal and subnormal codes alike, and the power-of-two FP16 multiply
// restores it exactly (|value| <= 448). NaN codes (0x7F, 0xFF) are not handled on that path:
// every producer quantizes with saturation to finite values.
__device__ __forceinline__ __half2 fp8_e4m3x2_to_half2(std::uint16_t storage) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
    const std::uint32_t s    = storage;
    const std::uint32_t bits = ((s & 0x007FU) << 7) | ((s & 0x0080U) << 8) | ((s & 0x7F00U) << 15) |
                               ((s & 0x8000U) << 16);
    __half2_raw raw;
    raw.x = static_cast<unsigned short>(bits);
    raw.y = static_cast<unsigned short>(bits >> 16);
    return __hmul2(__half2(raw), __half2half2(__ushort_as_half(0x5C00U))); // 256
#else
    __nv_fp8x2_e4m3 value;
    value.__x = storage;
    return static_cast<__half2>(value);
#endif
}

} // namespace ninfer::ops
