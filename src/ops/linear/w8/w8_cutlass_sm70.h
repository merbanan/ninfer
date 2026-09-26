#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops::detail {

// Wide-T W8G32_F16S projections on Volta tensor cores, the generic form of
// w8_gdn_input_cutlass_sm70: dequantize the weight into an FP16 scratch, cast the BF16
// activations, and run CUTLASS's Sm70 GEMM once per output segment. The only Volta W8 route
// the attention input projection and linear_add had was SIMT r8_c4 (about 2 TFLOP/s on an
// RTX 2060 SUPER against about 14 here), so from a few dozen columns on this wins even after
// paying for the dequant pass.
inline constexpr std::int32_t kW8CutlassMinCols = 64;

// One output segment: weight rows [row0, row0 + rows) into `out` [rows, cols] (column stride
// out_ld elements).
struct W8CutlassSegment {
    std::int32_t row0 = 0;
    std::int32_t rows = 0;
    void* out         = nullptr; // BF16
    std::int32_t out_ld = 0;
};

[[nodiscard]] std::size_t w8_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k,
                                                          std::int32_t cols);

// Whether `ws` has room for the route at this problem (callers fall back to SIMT otherwise).
[[nodiscard]] bool w8_cutlass_sm70_fits(const WorkspaceArena& ws, std::int32_t n, std::int32_t k,
                                        std::int32_t cols);

// out = W x (accumulate = false) or out += W x (accumulate = true) for every segment.
void w8_cutlass_sm70_run(const Tensor& x, const Weight& weight,
                         std::span<const W8CutlassSegment> segments, bool accumulate,
                         WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops::detail
