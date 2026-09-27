#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// Grouped routed-expert GEMMs of the Flash-Next prefill arm on Turing tensor cores
// (mma.m16n8k8, FP16 operands, FP32 accumulation). They read the grouping of the SIMT arm
// (expert_offsets/expert_counts/active_experts/grouped_tokens/grouped_paths) and write the same
// outputs: gate/up the BF16 SwiGLU activations [T][11][640], down the FP32 routed intermediate
// [T][2560][10] (route weights applied by the reduce kernel). Weights are decoded exactly to
// FP16 (E2M1 x E4M3 fits the FP16 significand) and the per-expert divisor is applied in FP32;
// the BF16 activations are rounded to FP16 (saturating), the only departure from the SIMT arm.
//
// The kernels live in their own sm_70 + sm_75 non-RDC archive; they exist only for sm_75.

// True on a compute capability 7.5 device unless NINFER_FLASH_NEXT_MOE_TURING=0 (read once).
[[nodiscard]] bool flash_next_moe_prefill_turing_enabled();
// Test hook: overrides the selection above for this process.
void flash_next_moe_prefill_turing_override(bool enabled);

struct FlashNextTuringGroups {
    const std::int32_t* expert_offsets;
    const std::int32_t* expert_counts;
    const std::int32_t* active_experts;
    const std::int32_t* active_count;
    const std::int32_t* grouped_tokens;
    const std::int32_t* grouped_paths;
    const std::int32_t* expert_slots; // nullable: bank index of each routed id
};

struct FlashNextTuringBank {
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    const float* divisors;
    std::uint64_t code_stride;
    std::uint64_t scale_stride;
};

void flash_next_moe_prefill_turing_gate_up(const FlashNextTuringGroups& groups, const FlashNextTuringBank& bank,
                                           const void* input_bf16, void* activations_bf16, cudaStream_t stream);

void flash_next_moe_prefill_turing_down(const FlashNextTuringGroups& groups, const FlashNextTuringBank& bank,
                                        const void* activations_bf16, float* down_intermediate, cudaStream_t stream);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
