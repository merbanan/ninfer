#pragma once

#include "ops/sparse_moe/cpu/sparse_moe_cpu.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// One NVFP4 expert matrix [rows, columns] in the artifact's expert-blockscale-k16-m128x4-v1
// encoding, read in place from host memory: row-major E2M1 codes (low nibble = even column),
// swizzled E4M3FN scales per 16 columns, and the FP32 weight divisor. A weight is
// e2m1(code) * e4m3(scale) / divisor, as the device kernels decode it.
struct HostNvfp4Matrix {
    const std::uint8_t* codes  = nullptr;
    const std::uint8_t* scales = nullptr;
    float divisor              = 1.0F;
    std::int32_t rows          = 0;
    std::int32_t columns       = 0;
};

// One routed expert applied to some token columns: gate/up [1280, 2560] (gate rows 0..639,
// up rows 640..1279) and down [2560, 640].
struct FlashNextColdExpert {
    HostNvfp4Matrix gate_up;
    HostNvfp4Matrix down;
    std::vector<std::int32_t> columns; // token columns routed to this expert
    std::vector<float> weights;        // route weight of each column
};

// Zeroes out[column * 2560 + row] for every column in [0, tokens) and adds
// weight * down(bf16(silu(gate(x)) * up(x))) of every work item, x being the BF16 column
// x_bf16[column * 2560 ...]. FP32 arithmetic with exact stored-weight decode; the SwiGLU value is
// rounded to BF16 before the down projection, as the device kernels store it.
void flash_next_cold_experts(std::span<const FlashNextColdExpert> work, const std::uint16_t* x_bf16,
                             std::int32_t tokens, float* out, ops::cpu::SpinPool& pool);

// Scalar reference decode of one stored weight (tests and diagnostics).
[[nodiscard]] float flash_next_nvfp4_weight(const HostNvfp4Matrix& matrix, std::int32_t row, std::int32_t column);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
