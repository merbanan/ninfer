#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct FlashNextMoeWorkspace {
    Tensor scores;
    Tensor ids;
    Tensor alpha;
    Tensor shared_scale;
    Tensor activations;
    // Prefill grouping & amortized GEMM workspace (tokens > 8)
    Tensor expert_counts;
    Tensor expert_offsets;
    Tensor active_expert_ids;
    Tensor active_count;
    Tensor grouped_tokens;
    Tensor grouped_paths;
    Tensor grouped_experts;
    Tensor token_to_pos;
    Tensor staged_down;
    Tensor down_intermediate;
    Tensor task_counter;
    // Prefill NVFP4 MMA activation quantization buffers (tokens > 8)
    Tensor act_codes;
    Tensor act_scales;
    Tensor down_act_codes;
    Tensor down_act_scales;
    // Packed BF16 [1280, T] = gate[640,T] | shared-act[640,T] for the MMA shared path.
    Tensor shared_gemm;
    // Decode arm only: FP32 [2560, T] routed-expert sum computed on the host for experts the
    // offload cache does not hold (their ids point at an all-zero slot); added before rounding.
    const float* cold_sum = nullptr;
    // Prefill arm, SIMT kernels only: bank index of each routed id [512] (device), so routed ids
    // stay < 512 for grouping while the weights come from any slot of a larger bank.
    const std::int32_t* expert_slots = nullptr;
};

// The SIMT arm processes prefill calls in slices of at most this many tokens, so its FP32 routed
// intermediate ([2560, 10, T], 105 MB per 1024 tokens) stays bounded for long prefill chunks.
inline constexpr std::int32_t kFlashNextSimtMoeTokens = 2048;

// True when prefill always takes the SIMT arm (the sm_70 build, which lacks the NVFP4 MMA
// route); its FP32 routed intermediate is then carved at every token count.
[[nodiscard]] bool flash_next_moe_prefill_simt_only() noexcept;

template <class Arena>
FlashNextMoeWorkspace allocate_flash_next_moe_workspace(Arena& arena, std::int32_t tokens) {
    FlashNextMoeWorkspace out;
    out.scores       = arena.alloc(DType::FP32, {513, tokens}, 256);
    out.ids          = arena.alloc(DType::I32, {10, tokens}, 16);
    out.alpha        = arena.alloc(DType::FP32, {10, tokens}, 16);
    out.shared_scale = arena.alloc(DType::FP32, {tokens}, 16);
    // Ten routed SwiGLU paths followed by the always-on shared path.
    out.activations = arena.alloc(DType::BF16, {640, 11, tokens}, 256);

    if (tokens > 8) {
        out.expert_counts     = arena.alloc(DType::I32, {512}, 16);
        out.expert_offsets    = arena.alloc(DType::I32, {513}, 16);
        out.active_expert_ids = arena.alloc(DType::I32, {512}, 16);
        out.active_count      = arena.alloc(DType::I32, {1}, 16);
        out.grouped_tokens    = arena.alloc(DType::I32, {10 * tokens}, 16);
        out.grouped_paths     = arena.alloc(DType::I32, {10 * tokens}, 16);
        out.grouped_experts   = arena.alloc(DType::I32, {10 * tokens}, 16);
        out.token_to_pos      = arena.alloc(DType::I32, {10 * tokens}, 16);
        if (tokens >= 512 && !flash_next_moe_prefill_simt_only()) {
            // MMA arm: BF16 staged routed outputs [2560, 10 * tokens], reduced in fixed order.
            // The envelope is computed by calling this with the chunk capacity, but a tail chunk
            // of 9..511 tokens takes the SIMT arm and carves an FP32 [2560, 10, T] intermediate,
            // which is larger than the BF16 staging when tokens < 1022. Pad the staging so the
            // envelope covers the worst SIMT tail (T = 511) at every capacity.
            constexpr std::int32_t kSimtTailColumns = 10 * 511 * 2; // FP32 [2560,10,511] in BF16 columns
            const std::int32_t staged_columns       = std::max(10 * tokens, kSimtTailColumns);
            out.staged_down   = arena.alloc(DType::BF16, {2'560, staged_columns}, 256);
        } else {
            const std::int32_t slice = flash_next_moe_prefill_simt_only()
                                           ? std::min(tokens, kFlashNextSimtMoeTokens)
                                           : tokens;
            out.down_intermediate = arena.alloc(DType::FP32, {2'560, 10, slice}, 256);
        }
        out.task_counter      = arena.alloc(DType::I32, {4}, 16);
        if (!flash_next_moe_prefill_simt_only()) {
            // NVFP4 activation codes and the packed shared GEMM input of the MMA arm.
            out.act_codes       = arena.alloc(DType::U8, {1'280, tokens}, 256);
            out.act_scales      = arena.alloc(DType::U8, {160, tokens}, 256);
            out.down_act_codes  = arena.alloc(DType::U8, {320, 11 * tokens}, 256);
            out.down_act_scales = arena.alloc(DType::U8, {40, 11 * tokens}, 256);
            out.shared_gemm     = arena.alloc(DType::BF16, {640, 2 * tokens}, 256);
        }
    }
    return out;
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
