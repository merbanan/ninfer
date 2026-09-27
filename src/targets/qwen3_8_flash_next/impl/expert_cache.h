#pragma once

#include "core/arena.h"
#include "ops/sparse_moe/cpu/sparse_moe_cpu.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"
#include "targets/qwen3_8_flash_next/impl/moe_workspace.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// Tuning of the host-resident expert path; defaults measured on an RTX 2060 SUPER (PCIe 3 x8).
struct FlashNextOffloadPolicy {
    // Device bytes of the expert cache; 0 takes the free device memory less `reserve_mib`.
    std::size_t cache_bytes = 0;
    std::size_t reserve_mib = 384;
    // Calls of at most this many tokens compute the experts the cache misses on the host
    // (0 disables: every miss is uploaded before use).
    std::int32_t host_max_tokens = 8;
    // Missing experts uploaded in the background per host-computed call, once an expert has
    // missed `admit_misses` times (halved every 64 decode calls of a layer).
    std::uint32_t promotions_per_call = 1;
    std::uint32_t admit_misses        = 4;
    // Prefill-uploaded experts enter the cache (least recently used slots are replaced).
    bool cache_prefill_experts = true;
    // Prefill-size calls of at most this many tokens split their missing experts between the
    // host and uploads by estimated cost (0 disables): an upload costs expert bytes / upload_gbps
    // on the bus plus pack_us_per_expert of host copying, a host expert host_us_per_expert plus
    // host_us_per_column per routed column after the first; the host takes the experts with the
    // fewest columns while that lowers max(host time, bus time).
    std::int32_t hybrid_max_tokens = 1024;
    double upload_gbps             = 6.5;
    double pack_us_per_expert      = 250.0;
    double host_us_per_expert      = 125.0;
    double host_us_per_column      = 22.0;
    // Host threads (the caller included); 0 = one per physical core.
    std::uint32_t host_threads = 0;
    bool report                = false;

    // NINFER_FLASH_NEXT_CACHE_MB, NINFER_FLASH_NEXT_CACHE_RESERVE_MB, NINFER_FLASH_NEXT_HOST_TOKENS,
    // NINFER_FLASH_NEXT_PROMOTE, NINFER_FLASH_NEXT_ADMIT, NINFER_FLASH_NEXT_CACHE_PREFILL,
    // NINFER_FLASH_NEXT_CPU_THREADS, NINFER_FLASH_NEXT_HYBRID, NINFER_FLASH_NEXT_HOST_US
    // ("<per expert>,<per column>"), NINFER_FLASH_NEXT_OFFLOAD_STATS.
    [[nodiscard]] static FlashNextOffloadPolicy from_environment();
};

struct FlashNextOffloadStats {
    std::uint64_t calls        = 0;
    std::uint64_t lookups      = 0; // distinct (call, expert) pairs
    std::uint64_t hits         = 0;
    std::uint64_t cold         = 0; // computed on the host
    std::uint64_t uploads      = 0; // uploaded before use
    std::uint64_t promotions   = 0; // uploaded in the background
    std::uint64_t inserted     = 0; // prefill uploads kept in the cache
    std::uint64_t hybrid_calls = 0; // prefill calls that sent experts to the host
    double host_seconds        = 0.0;
    double upload_seconds      = 0.0;
};

// Routed NVFP4 experts of the Flash-Next text layers left in the artifact mapping (host memory,
// normally the page cache over the file) behind a device cache of expert slots shared by all
// layers. The cache holds two NVFP4 banks (gate/up and down) in the artifact bank encoding with
// `slots` experts plus one all-zero expert at index `slots`.
//
// Decode-size calls run cached experts on the device and send the misses to the zero expert,
// computing them on the host into an FP32 addend the down kernel adds; a few misses are then
// uploaded in the background (copy stream) so the cache follows the routing. Prefill-size calls
// assemble their active experts in the runtime's compact staging banks (device copies of cached
// ones, uploads of the rest) and remap the routed ids to staging indices, which keeps the
// grouped prefill kernels unchanged.
class FlashNextExpertCache {
public:
    explicit FlashNextExpertCache(FlashNextOffloadPolicy policy = FlashNextOffloadPolicy::from_environment());
    ~FlashNextExpertCache();

    FlashNextExpertCache(const FlashNextExpertCache&)            = delete;
    FlashNextExpertCache& operator=(const FlashNextExpertCache&) = delete;

    // The routed part of flash_next_moe after routing: scratch holds ids/alpha/shared_scale.
    void run(const Tensor& input, const MoeWeights& weights, FlashNextMoeWorkspace& scratch, Tensor& output,
             cudaStream_t stream, void* staging, std::size_t staging_bytes);

    [[nodiscard]] const FlashNextOffloadStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const FlashNextOffloadPolicy& policy() const noexcept { return policy_; }
    // Prints the counters to stderr (NINFER_FLASH_NEXT_OFFLOAD_STATS=1 does so at exit).
    void report() const;

private:
    // Byte geometry of one NVFP4 bank.
    struct Bank {
        std::size_t code_bytes = 0, scale_bytes = 0; // per expert
        std::size_t scale_offset = 0, divisor_offset = 0, total = 0;
        std::int32_t rows = 0, columns = 0;
    };
    struct Slot {
        std::int64_t key        = -1; // layer * 512 + expert
        std::uint64_t last_used = 0;
        bool pending            = false;
    };
    struct Promotion {
        std::uint32_t slot   = 0;
        std::int64_t key     = 0;
        std::size_t buffer   = 0;
    };

    void allocate(const MoeWeights& weights);
    [[nodiscard]] int layer_of(const MoeWeights& weights);
    void decode(int layer, const Tensor& input, const MoeWeights& weights, FlashNextMoeWorkspace& scratch,
                Tensor& output, cudaStream_t stream);
    void prefill(int layer, const Tensor& input, const MoeWeights& weights, FlashNextMoeWorkspace& scratch,
                 Tensor& output, cudaStream_t stream, void* staging, std::size_t staging_bytes);
    void retire_promotions();
    [[nodiscard]] std::int64_t victim() const;
    // Copies `experts` of both host banks into `dst`, packed_bytes() apart (gate codes, scales,
    // divisor, down codes, scales, divisor, each 256-aligned), on the host pool.
    void pack(const MoeWeights& weights, std::span<const std::int32_t> experts, std::byte* dst);
    // Forgets the expert held by `slot` (the slot stays allocated to the caller).
    void evict(std::uint32_t slot);
    [[nodiscard]] std::size_t packed_bytes() const noexcept;
    // Six copies of one packed expert into expert `index` of the banks at gate_base/down_base.
    void unpack(const std::byte* src, std::byte* gate_base, std::byte* down_base, std::int64_t index,
                cudaMemcpyKind kind, cudaStream_t stream) const;
    // Copies expert `from` of banks (src_gate, src_down) to expert `to` of (dst_gate, dst_down).
    void copy_expert(const std::byte* src_gate, const std::byte* src_down, const Bank& src_gb,
                     const Bank& src_db, std::int64_t from, std::byte* dst_gate, std::byte* dst_down,
                     const Bank& dst_gb, const Bank& dst_db, std::int64_t to, cudaStream_t stream) const;

    FlashNextOffloadPolicy policy_;
    bool allocated_ = false;
    std::uint32_t slots_ = 0;
    Bank gate_, down_;
    DeviceBuffer gate_pool_, down_pool_;
    std::vector<Slot> slot_;
    std::unordered_map<std::int64_t, std::uint32_t> resident_;
    std::unordered_map<const std::byte*, int> layers_;
    std::vector<std::uint16_t> miss_counts_;
    std::vector<std::uint64_t> layer_calls_;
    std::uint64_t clock_ = 0;
    std::vector<std::uint8_t> needed_; // per slot, this call
    cudaStream_t last_stream_ = nullptr;

    PinnedHostBuffer host_ids_{sizeof(std::int32_t) * 10 * 8};
    PinnedHostBuffer host_alpha_{sizeof(float) * 10 * 8};
    PinnedHostBuffer host_x_{sizeof(std::uint16_t) * 2560 * 8};
    PinnedHostBuffer host_cold_{sizeof(float) * 2560 * 8};
    DeviceBuffer device_cold_{sizeof(float) * 2560 * 8};
    std::unique_ptr<PinnedHostBuffer> prefill_ids_, prefill_x_, prefill_alpha_, prefill_cold_;
    DeviceBuffer prefill_cold_device_;

    // Prefill uploads: batches of packed experts in two alternating pinned buffers.
    static constexpr std::size_t kBatchExperts = 8;
    std::array<std::unique_ptr<PinnedHostBuffer>, 2> batch_;
    std::array<cudaEvent_t, 2> batch_free_{};
    // Background promotions: one pinned buffer per upload in flight.
    static constexpr std::size_t kPromotionBuffers = 4;
    std::array<std::unique_ptr<PinnedHostBuffer>, kPromotionBuffers> promotion_buffer_;
    std::array<cudaEvent_t, kPromotionBuffers> promotion_done_{};
    std::array<bool, kPromotionBuffers> promotion_busy_{};
    std::vector<Promotion> promotions_;
    cudaStream_t copy_stream_ = nullptr;
    cudaEvent_t compute_mark_ = nullptr;

    std::unique_ptr<ops::cpu::SpinPool> cpu_pool_;
    FlashNextOffloadStats stats_;
};

// The process-wide cache flash_next_moe uses for host-resident expert banks, created on first
// use; nullptr when NINFER_FLASH_NEXT_OFFLOAD=0 selects the per-call staging path instead.
[[nodiscard]] FlashNextExpertCache* flash_next_expert_cache();

} // namespace ninfer::targets::qwen3_8_flash_next::detail
