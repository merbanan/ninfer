#pragma once

// Turing-native (sm_75) tensor-core step for the fused-dequant Q4/Q5 GEMM tiles
// (q4_volta_mma_gemm.cuh, q5_volta_mma_gemm.cuh). Those tiles stage activations as
// x_sh[token][k] and decoded weights as w_sh[warp][row][k] in FP16 and feed Volta's
// mma.m8n8k4, which Turing runs at a fraction of its native rate (measured 12.7 against
// 20.5 TFLOP/s for mma.m16n8k8 on an RTX 2060 SUPER). This header consumes the same shared
// tiles with mma.m16n8k8: one warp computes its 32 tokens x 8 rows as two 16 x 8 tiles.
//
// Fragments (g = lane / 4, q = lane % 4): A row g and g + 8, k pair 2q; B row (output column)
// g, k pair 2q; C rows g (c0, c1) and g + 8 (c2, c3), columns 2q and 2q + 1. Every operand is
// one 32-bit shared load; with the tiles' 8-half row padding (a 20-word stride) the eight rows
// a load touches fall in distinct banks.

#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750

__device__ __forceinline__ void turing_mma_m16n8k8(float (&c)[4], std::uint32_t a0, std::uint32_t a1,
                                                   std::uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}

// One staged K step: d[mt] += x rows [16 mt, 16 mt + 16) x w rows [0, 8), over KStep columns.
template <int KStep, int Stride>
__device__ __forceinline__ void turing_tile_mma_step(float (&d)[2][4], const __half* x_sh, const __half* w_sh) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int g    = lane >> 2;
    const int q    = lane & 3;
#pragma unroll
    for (int kk = 0; kk < KStep; kk += 8) {
        const std::uint32_t b0 = *reinterpret_cast<const std::uint32_t*>(w_sh + g * Stride + kk + 2 * q);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const __half* rows     = x_sh + (mt * 16 + g) * Stride + kk + 2 * q;
            const std::uint32_t a0 = *reinterpret_cast<const std::uint32_t*>(rows);
            const std::uint32_t a1 = *reinterpret_cast<const std::uint32_t*>(rows + 8 * Stride);
            turing_mma_m16n8k8(d[mt], a0, a1, b0);
        }
    }
}

// Token row and output column of accumulator element i of tile mt.
__device__ __forceinline__ int turing_tile_row(int mt, int i) {
    return mt * 16 + ((static_cast<int>(threadIdx.x) & 31) >> 2) + (i >= 2 ? 8 : 0);
}
__device__ __forceinline__ int turing_tile_col(int i) {
    return 2 * (static_cast<int>(threadIdx.x) & 3) + (i & 1);
}

#endif

} // namespace ninfer::ops::detail
