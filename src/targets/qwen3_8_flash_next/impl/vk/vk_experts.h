#pragma once

#include "targets/qwen3_8_flash_next/impl/model_view.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// Tuning of the RX 570 (Vulkan) expert tier; defaults measured on an AMD Radeon RX 570 (Polaris10,
// 4 GB, ~1.2-1.5 GB free) reached over PCIe 3.0 x8, own lanes from the CUDA device.
struct FlashNextVkPolicy {
    bool enabled              = true; // NINFER_FLASH_NEXT_VK (0 disables; default on if an AMD
                                       // Vulkan device is found)
    std::int32_t device_index = -1;   // NINFER_FLASH_NEXT_VK_DEVICE: physical device index among
                                       // vendorID 0x1002 devices (-1: first one)
    std::size_t cache_bytes   = 0;    // NINFER_FLASH_NEXT_VK_CACHE_MB; 0 takes the reported
                                       // VK_EXT_memory_budget less reserve_mib
    std::size_t reserve_mib   = 256;  // NINFER_FLASH_NEXT_VK_RESERVE_MB
    std::uint32_t promote_per_call = 2; // NINFER_FLASH_NEXT_VK_PROMOTE
    bool report                    = false; // NINFER_FLASH_NEXT_OFFLOAD_STATS

    [[nodiscard]] static FlashNextVkPolicy from_environment();
};

struct FlashNextVkStats {
    std::uint64_t calls      = 0;
    std::uint64_t hits       = 0; // distinct (call, expert) pairs resolved on the RX 570
    std::uint64_t promotions = 0;
    double submit_seconds    = 0.0; // submit() -> wait() wall time, summed
};

// One RX 570 job of the current decode call: an expert resident in the Vulkan slot pool, with the
// token columns of this call routed to it and their route weights (mirrors FlashNextColdExpert,
// but referencing a Vulkan slot instead of the host mapping).
struct FlashNextVkJob {
    std::uint32_t slot = 0;
    std::vector<std::int32_t> columns;
    std::vector<float> weights;
};

// The RX 570 expert tier between the CUDA cache and the CPU (decode calls, T <= 8, only): experts
// resident here are computed on the second GPU while the CPU handles the rest of the call's
// misses; both host FP32 addends are summed by the caller (FlashNextExpertCache::decode()) before
// the down kernel adds them to the device MoE output. Single dispatch pair (gate_up, down) per
// call regardless of how many resident experts it uses.
//
// Never throws once constructed: any Vulkan failure during create() is reported and yields
// nullptr; a lost device during use is not handled (the process restarts, like a CUDA context
// loss elsewhere in NInfer).
class FlashNextVkExperts {
public:
    // Returns nullptr if disabled, no usable AMD Vulkan device is found, or initialization fails
    // for any reason (logs one line to stderr in that case; callers run without the tier).
    static std::unique_ptr<FlashNextVkExperts> create(const MoeWeights& weights, FlashNextVkPolicy policy);
    ~FlashNextVkExperts();

    FlashNextVkExperts(const FlashNextVkExperts&)            = delete;
    FlashNextVkExperts& operator=(const FlashNextVkExperts&) = delete;

    [[nodiscard]] std::size_t packed_bytes() const noexcept { return packed_bytes_; }
    [[nodiscard]] std::uint32_t slot_count() const noexcept { return slots_; }
    [[nodiscard]] const FlashNextVkPolicy& policy() const noexcept { return policy_; }
    [[nodiscard]] bool resident(std::int64_t key) const;
    // Slot of a resident key; only valid when resident(key) is true.
    [[nodiscard]] std::uint32_t slot_of(std::int64_t key) const;

    [[nodiscard]] const FlashNextVkStats& stats() const noexcept { return stats_; }
    // Prints the counters to stderr (NINFER_FLASH_NEXT_OFFLOAD_STATS=1 does so at exit, alongside
    // FlashNextExpertCache::report()).
    void report() const;

    // Async: uploads x and the job table and dispatches gate_up then down. `jobs` must not be
    // empty; every job's slot must currently be resident. `tokens` is the call's token count
    // (<= 8); x_bf16 is [tokens][2560] (row-major, BF16), widened to FP32 on the host before
    // upload (the device has no FP16 arithmetic).
    void submit(std::span<const FlashNextVkJob> jobs, const std::uint16_t* x_bf16, std::int32_t tokens);
    // Blocks for the dispatch submitted by submit() to finish, then adds the resulting FP32
    // [tokens][2560] addend into `accumulate_into` (token-major, out[t*2560+row] += ...). The
    // caller zeroes accumulate_into first when nothing else already wrote it this call.
    void wait(float* accumulate_into, std::int32_t tokens);

    // Promotion: `packed` is one expert already in FlashNextExpertCache::pack()'s byte layout
    // (packed_bytes() long: 256-aligned gate codes, gate scales, gate divisor, down codes, down
    // scales, down divisor). Copies it into a staging buffer and starts an async upload into a
    // free or LRU slot; poll retire_promotions() (checked at the start of the next call) to see
    // it become resident. A no-op if no promotion buffer is free or every slot is in flight.
    void begin_promotion(std::int64_t key, const std::byte* packed);
    void retire_promotions();
    [[nodiscard]] bool promotion_in_flight(std::int64_t key) const;
    [[nodiscard]] bool has_free_promotion_slot() const;

private:
    struct Impl;
    explicit FlashNextVkExperts(std::unique_ptr<Impl> impl, FlashNextVkPolicy policy, std::size_t packed_bytes,
                               std::uint32_t slots);

    std::unique_ptr<Impl> impl_;
    FlashNextVkPolicy policy_;
    std::size_t packed_bytes_ = 0;
    std::uint32_t slots_      = 0;
    FlashNextVkStats stats_;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
