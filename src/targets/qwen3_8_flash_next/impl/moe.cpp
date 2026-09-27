#include "targets/qwen3_8_flash_next/impl/moe.h"

#include "targets/qwen3_8_flash_next/impl/expert_cache.h"

#include "core/layout.h"
#include "targets/qwen3_8_flash_next/impl/moe_kernels.h"
#include "targets/qwen3_8_flash_next/impl/moe_route.h"
#include "targets/qwen3_8_flash_next/impl/moe_workspace.h"
#include "targets/qwen3_8_flash_next/impl/stage_ledger.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "core/device.h"

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool exact_bf16_weight(const Weight& weight, std::int32_t rows, std::int32_t columns) {
    return weight.qtype == QType::BF16_CTRL && weight.layout == QuantLayout::Contiguous &&
           weight.n == rows && weight.k == columns && weight.ndim == 2 && weight.shape[0] == rows &&
           weight.shape[1] == columns && weight.padded_shape[0] == rows &&
           weight.padded_shape[1] == columns && weight.qdata == weight.payload &&
           weight.payload_bytes >= static_cast<std::uint64_t>(rows) * columns * 2 &&
           aligned_to(weight.qdata, 16);
}

bool exact_expert_bank(const Nvfp4ExpertBankView& bank, std::int32_t rows, std::int32_t columns) {
    const std::uint64_t elements = static_cast<std::uint64_t>(rows) * columns;
    return bank.experts > 0 && bank.experts <= 512 && bank.rows == rows && bank.columns == columns &&
           bank.code_bytes_per_expert == elements / 2 &&
           bank.scale_bytes_per_expert == elements / 16 && aligned_to(bank.codes, 16) &&
           aligned_to(bank.scales, 16) && aligned_to(bank.weight_scale_divisors, 16);
}


bool exact_bf16_expert_bank(const Bf16ExpertBankView& bank, std::int32_t rows, std::int32_t columns) {
    const std::uint64_t elements = static_cast<std::uint64_t>(rows) * columns;
    return bank.experts == 512 && bank.rows == rows && bank.columns == columns &&
           bank.bytes_per_expert == elements * sizeof(std::uint16_t) &&
           aligned_to(bank.data, 16);
}

} // namespace

// The process-wide expert cache of the host-resident banks (one runtime per process). It is never
// destroyed: its device and pinned buffers go with the process, after the optional report.
namespace {
FlashNextExpertCache* g_expert_cache = nullptr;
} // namespace

FlashNextExpertCache* flash_next_expert_cache() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_FLASH_NEXT_OFFLOAD");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    if (!enabled) { return nullptr; }
    if (g_expert_cache == nullptr) {
        g_expert_cache = new FlashNextExpertCache();
        if (g_expert_cache->policy().report) {
            std::atexit([] { g_expert_cache->report(); });
        }
    }
    return g_expert_cache;
}

std::size_t flash_next_moe_workspace_capacity_bytes(std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("Flash-Next MoE workspace requires positive tokens");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_flash_next_moe_workspace(layout, max_tokens);
    return layout.peak_bytes(256);
}

void flash_next_moe(const Tensor& input, const MoeWeights& weights, Tensor& output,
                    WorkspaceArena& workspace, cudaStream_t stream,
                    void* expert_staging, std::size_t expert_staging_bytes) {
    const std::int32_t tokens = input.ne[1];
    if (input.dtype != DType::BF16 || output.dtype != DType::BF16 || input.ne[0] != 2'560 ||
        output.ne[0] != 2'560 || tokens < 1 || output.ne[1] != tokens ||
        input.ne[2] != 1 || input.ne[3] != 1 || output.ne[2] != 1 || output.ne[3] != 1 ||
        !input.is_contiguous() || !output.is_contiguous() || !aligned_to(input.data, 16) ||
        !aligned_to(output.data, 16) || !exact_bf16_weight(weights.router, 512, 2'560) ||
        !exact_bf16_weight(weights.shared_down, 2'560, 640) ||
        !exact_bf16_weight(weights.shared_gate, 640, 2'560) ||
        !exact_bf16_weight(weights.shared_up, 640, 2'560) ||
        !exact_bf16_weight(weights.shared_gate_weight, 1, 2'560) ||
        !exact_expert_bank(weights.expert_gate_up, 1'280, 2'560) ||
        !exact_expert_bank(weights.expert_down, 2'560, 640) || stream == nullptr) {
        throw std::invalid_argument("Flash-Next MoE received an invalid exact target view");
    }
    const auto scope              = workspace.scope();
    FlashNextMoeWorkspace scratch = allocate_flash_next_moe_workspace(workspace, tokens);
    flash_next_route(input, weights.router, weights.shared_gate_weight, scratch.scores, scratch.ids,
                     scratch.alpha, scratch.shared_scale, stream);
    stage_ledger_record(stream, FlashNextStageId::MoE_Router);
#if defined(NINFER_VOLTA_BUILD)
    if (weights.expert_gate_up.mapped_host != weights.expert_down.mapped_host) {
        throw std::runtime_error("Flash-Next Volta expert banks must use matching residency");
    }
    if (weights.expert_gate_up.mapped_host) {
        if (auto* cache = flash_next_expert_cache(); cache != nullptr) {
            cache->run(input, weights, scratch, output, stream);
            return;
        }
        std::vector<std::int32_t> host_ids(static_cast<std::size_t>(tokens) * 10);
        CUDA_CHECK(cudaMemcpyAsync(host_ids.data(), scratch.ids.data,
                                   static_cast<std::size_t>(tokens) * 10 * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<std::int32_t> active(host_ids.begin(), host_ids.end());
        const int expert_count = weights.expert_gate_up.experts;
        if (expert_count <= 0 || weights.expert_down.experts != expert_count || expert_count > 512) {
            throw std::runtime_error("Flash-Next Volta expert banks have inconsistent expert counts");
        }
        for (const auto expert : active) {
            if (expert < 0 || expert >= expert_count) {
                throw std::runtime_error("Flash-Next Volta router produced an out-of-range expert id");
            }
        }
        std::sort(active.begin(), active.end());
        active.erase(std::unique(active.begin(), active.end()), active.end());
        if (active.empty() || active.size() > static_cast<std::size_t>(expert_count)) {
            throw std::runtime_error("Flash-Next Volta routing produced an invalid expert set");
        }
        // Stage only selected experts into a compact bank and remap routed ids to slots.
        // Plane-wise copies preserve the NVFP4 bank encoding expected by make_nvfp4_expert_bank_view.
        const int slots = static_cast<int>(active.size());
        struct DecodeExpertCache {
            const std::byte* gate_source = nullptr;
            const std::byte* down_source = nullptr;
            void* staging_source = nullptr;
            std::vector<std::int32_t> experts;
            std::size_t gate_bytes = 0;
            std::size_t down_bytes = 0;
        };
        static thread_local DecodeExpertCache cache;
        if (expert_staging == nullptr || expert_staging_bytes == 0) {
            throw std::runtime_error("Flash-Next Volta mapped experts require runtime-owned staging storage");
        }
        if (weights.expert_gate_up.mapped_payload == nullptr ||
            weights.expert_down.mapped_payload == nullptr ||
            weights.expert_gate_up.mapped_payload_bytes == 0 ||
            weights.expert_down.mapped_payload_bytes == 0) {
            throw std::runtime_error("Flash-Next Volta mapped expert payload is unavailable");
        }

        const auto bank_bytes = [slots](const Nvfp4ExpertBankView& bank) {
            return static_cast<std::size_t>(bank.code_bytes_per_expert) * slots +
                   static_cast<std::size_t>(bank.scale_bytes_per_expert) * slots +
                   sizeof(float) * slots;
        };
        const std::size_t gate_required = bank_bytes(weights.expert_gate_up);
        const std::size_t down_required = bank_bytes(weights.expert_down);
        const std::size_t down_offset = (gate_required + 255U) & ~std::size_t{255U};
        if (down_offset > expert_staging_bytes ||
            down_required > expert_staging_bytes - down_offset) {
            throw std::runtime_error("Flash-Next Volta expert staging reservation is too small");
        }
        auto* staging_base = static_cast<std::byte*>(expert_staging);
        void* gate_storage = staging_base;
        void* down_storage = staging_base + down_offset;

        const bool cache_hit = cache.gate_source == weights.expert_gate_up.mapped_payload &&
                               cache.down_source == weights.expert_down.mapped_payload &&
                               cache.staging_source == expert_staging &&
                               cache.experts == active;
        const auto stage_bank = [&](const Nvfp4ExpertBankView& host_bank, void* storage) {
            const std::size_t code_bytes = static_cast<std::size_t>(host_bank.code_bytes_per_expert) * slots;
            const std::size_t scale_bytes = static_cast<std::size_t>(host_bank.scale_bytes_per_expert) * slots;
            const std::size_t required = code_bytes + scale_bytes + sizeof(float) * slots;
            auto* base = static_cast<std::byte*>(storage);
            for (int slot = 0; slot < slots; ++slot) {
                const int expert = active[slot];
                CUDA_CHECK(cudaMemcpyAsync(base + static_cast<std::size_t>(slot) * host_bank.code_bytes_per_expert,
                    host_bank.codes + static_cast<std::size_t>(expert) * host_bank.code_bytes_per_expert,
                    host_bank.code_bytes_per_expert, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(base + code_bytes + static_cast<std::size_t>(slot) * host_bank.scale_bytes_per_expert,
                    host_bank.scales + static_cast<std::size_t>(expert) * host_bank.scale_bytes_per_expert,
                    host_bank.scale_bytes_per_expert, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(base + code_bytes + scale_bytes + static_cast<std::size_t>(slot) * sizeof(float),
                    host_bank.weight_scale_divisors + expert, sizeof(float), cudaMemcpyHostToDevice, stream));
            }
            return required;
        };
        if (!cache_hit) {
            cache.gate_bytes = stage_bank(weights.expert_gate_up, gate_storage);
            cache.down_bytes = stage_bank(weights.expert_down, down_storage);
            cache.gate_source = weights.expert_gate_up.mapped_payload;
            cache.down_source = weights.expert_down.mapped_payload;
            cache.staging_source = expert_staging;
            cache.experts = active;
        }
        std::array<std::int32_t, 512> slot_of{};
        slot_of.fill(-1);
        for (int slot = 0; slot < slots; ++slot) slot_of[active[slot]] = slot;
        for (int i = 0; i < tokens * 10; ++i) host_ids[i] = slot_of[host_ids[i]];
        CUDA_CHECK(cudaMemcpyAsync(scratch.ids.data, host_ids.data(),
                                   static_cast<std::size_t>(tokens) * 10 * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream));

        MoeWeights staged = weights;
        staged.expert_gate_up = make_nvfp4_expert_bank_view(gate_storage, cache.gate_bytes, slots, 1'280, 2'560);
        staged.expert_down = make_nvfp4_expert_bank_view(down_storage, cache.down_bytes, slots, 2'560, 640);
        flash_next_moe_kernels_launch(input, staged, scratch, output, stream);
        // Runtime-owned staging has allocation lifetime, so kernels may complete asynchronously.
        return;
    }
#endif
    flash_next_moe_kernels_launch(input, weights, scratch, output, stream);

    // Diagnostic NINFER_FLASH_NEXT_TRACE_ROUTING only; not on the default prefill chunk path.
    static const char* trace_routing_env = std::getenv("NINFER_FLASH_NEXT_TRACE_ROUTING");
    if (trace_routing_env != nullptr && trace_routing_env[0] != '\0' &&
        tokens >= 512) {
        static int s_route_call = 0;
        int h_active = 0;
        cudaMemcpyAsync(&h_active, scratch.active_count.data, sizeof(int), cudaMemcpyDeviceToHost, stream);
        std::vector<int> h_counts(512);
        cudaMemcpyAsync(h_counts.data(), scratch.expert_counts.data, 512 * sizeof(int), cudaMemcpyDeviceToHost, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_grp = 0, min_grp = 999999, sum_grp = 0;
        for (int e = 0; e < 512; ++e) {
            if (h_counts[e] > 0) {
                max_grp = std::max(max_grp, h_counts[e]);
                min_grp = std::min(min_grp, h_counts[e]);
                sum_grp += h_counts[e];
            }
        }
        int layer_idx = (s_route_call++) % 48;
        if (layer_idx == 0) {
            std::fprintf(stderr, "\n--- MoE Routing Trace (T=%d) ---\n", tokens);
            std::fprintf(stderr, "Layer | Active Experts | %% Active | Min Group | Avg Group | Max Group\n");
            std::fprintf(stderr, "------+----------------+----------+-----------+-----------+----------\n");
        }
        std::fprintf(stderr, " L%02d  |   %3d / 512    |  %5.1f%%  |    %3d    |   %5.1f   |    %3d   \n",
                     layer_idx, h_active, h_active * 100.0f / 512.0f, min_grp, (float)sum_grp / h_active, max_grp);
        if (layer_idx == 47) {
            std::fprintf(stderr, "-----------------------------------------------------------------\n\n");
        }
    }
}

void flash_next_moe_bf16(const Tensor& input, const MoeBf16Weights& weights, Tensor& output,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = input.ne[1];
    if (input.dtype != DType::BF16 || output.dtype != DType::BF16 || input.ne[0] != 2'560 ||
        output.ne[0] != 2'560 || tokens < 1 || output.ne[1] != tokens ||
        input.ne[2] != 1 || input.ne[3] != 1 || output.ne[2] != 1 || output.ne[3] != 1 ||
        !input.is_contiguous() || !output.is_contiguous() || !aligned_to(input.data, 16) ||
        !aligned_to(output.data, 16) || !exact_bf16_weight(weights.router, 512, 2'560) ||
        !exact_bf16_weight(weights.shared_down, 2'560, 640) ||
        !exact_bf16_weight(weights.shared_gate, 640, 2'560) ||
        !exact_bf16_weight(weights.shared_up, 640, 2'560) ||
        !exact_bf16_weight(weights.shared_gate_weight, 1, 2'560) ||
        !exact_bf16_expert_bank(weights.expert_gate_up, 1'280, 2'560) ||
        !exact_bf16_expert_bank(weights.expert_down, 2'560, 640) || stream == nullptr) {
        throw std::invalid_argument("Flash-Next BF16 MoE received an invalid exact target view");
    }
    const auto scope              = workspace.scope();
    FlashNextMoeWorkspace scratch = allocate_flash_next_moe_workspace(workspace, tokens);
    flash_next_route(input, weights.router, weights.shared_gate_weight, scratch.scores, scratch.ids,
                     scratch.alpha, scratch.shared_scale, stream);
    flash_next_moe_bf16_kernels_launch(input, weights, scratch, output, stream);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
