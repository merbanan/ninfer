#pragma once

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/host_worker_pool.h"
#include "core/tensor.h"
#include "ninfer/ops/sparse_moe.h"
#include "ops/sparse_moe/cpu/sparse_moe_cpu.h"

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace ninfer::targets::qwen3_6_35b_a3b::detail {

inline constexpr std::size_t kOffloadTextLayers = 40;
inline constexpr std::int32_t kOffloadExperts   = 256;

// One layer's routed banks inside the artifact file mapping (row-split-k128-v1 planes).
struct HostRoutedBanks {
    artifact::NumericFormat down_format = artifact::NumericFormat::Q5G64_F16S;
    std::span<const std::byte> gate_up; // Q4G64_F16S [256*1024, 2048]
    std::span<const std::byte> down;    // Q5G64_F16S or Q6G64_F16S [256*2048, 512]
};

// Tuning of the host/device split. The defaults come from the measurements in docs/v100.md.
struct ExpertOffloadPolicy {
    // Decode and small-T calls compute missing experts on the host instead of uploading them.
    bool host_cold_experts = true;
    // Missing experts per call uploaded in the background (a separate copy stream) so the device
    // cache keeps adapting; they are computed on the host for the call that missed them.
    std::uint32_t promotions_per_call = 1;
    // Threads of the host expert pool (the caller included); 0 selects one per physical core
    // (half the hardware threads), which measured best (SMT threads only add contention).
    std::uint32_t host_threads = 0;
    // A missing expert is promoted only once it has missed this many times (decayed by half
    // every 64 decode tokens); 1 admits every miss.
    std::uint32_t admit_misses = 4;
    // Prefill calls of at least this many tokens upload the next layer's non-resident experts in
    // the background while the current layer computes (0 disables). Below a few hundred tokens the
    // bulk queue delays the experts the next layer actually selects (measured: a 91-token prompt
    // prefilled at 23 instead of 31 tok/s with a 64-token threshold).
    std::uint32_t prefetch_min_tokens = 256;

    // Defaults, overridden by NINFER_OFFLOAD_COLD, NINFER_OFFLOAD_PROMOTE, NINFER_OFFLOAD_CPU_THREADS,
    // NINFER_OFFLOAD_ADMIT, NINFER_OFFLOAD_PREFETCH.
    [[nodiscard]] static ExpertOffloadPolicy from_environment();
};

struct ExpertOffloadStats {
    std::uint64_t calls         = 0;
    std::uint64_t lookups       = 0; // distinct (call, expert) pairs
    std::uint64_t hits          = 0; // resident on the device
    std::uint64_t cold          = 0; // computed on the host
    std::uint64_t uploads       = 0; // uploaded before use (prefill-size calls)
    std::uint64_t promotions    = 0; // uploaded in the background
    std::uint64_t prefetched    = 0; // uploaded ahead of their layer during prefill
    std::uint64_t prefetch_used = 0; // prefetched experts the next layer selected
    std::uint64_t bytes_copied  = 0;
    double upload_seconds       = 0.0; // host time staging uploads on the critical path
    double host_compute_seconds = 0.0;
    double prefill_moe_seconds  = 0.0; // NINFER_OFFLOAD_TIMING only
};

// Routed experts of the 35B-A3B Text layers held in host memory (the artifact file mapping, so
// normally the page cache) with a device LRU cache of expert slots. Layers whose routed down bank
// has the same format share one slot pool; a pool is a pair of row-split banks with `slots`
// experts in the registered per-expert row geometry plus one all-zero bank, which SparseMoe reads
// through its SparseMoeExpertResidency.
//
// Prefill-size calls upload every missing expert before use (two pinned staging buffers, host
// copy of one expert overlapping the upload of the previous one). Decode and small-T calls
// instead compute missing experts on the host from the mapping while the device runs the
// resident ones, and upload a few of them in the background on a separate copy stream so the
// cache follows the routing. All device reads of a slot are ordered before its overwrite: a
// foreground upload is on the caller's stream, a background one waits on an event recorded there.
class ExpertOffload {
public:
    ExpertOffload(std::array<HostRoutedBanks, kOffloadTextLayers> banks, std::size_t requested_bytes,
                  std::int32_t max_tokens, ExpertOffloadPolicy policy = ExpertOffloadPolicy::from_environment());
    ~ExpertOffload();

    ExpertOffload(const ExpertOffload&)            = delete;
    ExpertOffload& operator=(const ExpertOffload&) = delete;

    // Minimum device bytes: 256 slots (one full layer, which a long prefill chunk can touch) plus
    // the zero bank per down format.
    [[nodiscard]] std::size_t minimum_device_bytes() const noexcept;

    // Allocates the slot pools from `budget_bytes` (the requested size if one was given).
    void allocate(std::size_t budget_bytes);
    [[nodiscard]] bool allocated() const noexcept { return allocated_pools_; }
    [[nodiscard]] std::size_t requested_bytes() const noexcept { return requested_bytes_; }

    // SparseMoe of Text layer `layer` with its routed experts served from the cache.
    void sparse_moe(std::size_t layer, const Tensor& x, const ops::SparseMoeWeights& resident,
                    Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream);

    [[nodiscard]] const ExpertOffloadStats& stats() const noexcept { return stats_; }

private:
    struct Slot {
        std::int64_t key        = -1; // layer * 256 + expert, or -1 when empty
        std::uint64_t last_used = 0;
        bool pending            = false; // background upload not yet complete
    };
    // Byte geometry of one expert inside host banks and device pools.
    struct PlaneLayout {
        std::size_t gate_code_row = 0, gate_scale_row = 0;
        std::size_t down_code_row = 0, down_high_row = 0, down_scale_row = 0;
        [[nodiscard]] std::size_t expert_bytes() const noexcept;
    };
    // Offsets of the planes inside one layer's host banks.
    struct HostPlanes {
        std::size_t gate_scale_offset = 0, down_high_offset = 0, down_scale_offset = 0;
    };
    struct Pool {
        artifact::NumericFormat down_format = artifact::NumericFormat::Q5G64_F16S;
        std::uint32_t layers                = 0;
        PlaneLayout planes;
        std::vector<Slot> slots; // the zero bank is index slots.size()
        DeviceBuffer gate_codes, gate_scales, down_codes, down_high, down_scales;
        Weight gate_up, down;
    };
    struct Promotion {
        std::size_t pool   = 0;
        std::uint32_t slot = 0;
        std::size_t buffer = 0;
    };
    // One expert the prefetch thread uploads ahead of its layer.
    struct PrefetchJob {
        std::size_t pool   = 0;
        std::uint32_t slot = 0;
        std::size_t layer  = 0;
        std::int32_t expert = 0;
        cudaEvent_t done    = nullptr;
        bool issued         = false; // guarded by prefetch_mutex_
    };

    void acquire(std::size_t layer, std::span<const std::int32_t> selected, std::int32_t* bank_of_expert,
                 bool cold_allowed, cudaStream_t stream);
    void cold_compute(std::size_t layer, std::span<const std::int32_t> selected, const float* alpha,
                      const std::uint16_t* x_bf16, std::int32_t tokens, const std::int32_t* bank_of_expert,
                      float* cold_sum);
    void retire_promotions();
    void plan_prefetch(std::size_t layer, std::size_t pool_index);
    // Waits until the prefetch of `key` is issued and orders `stream` after its upload.
    void claim_prefetch(std::int64_t key, cudaStream_t stream);
    void retire_prefetches();
    void prefetch_loop();
    // Least recently used slot outside `needed_` that has no upload in flight, or -1.
    [[nodiscard]] std::int64_t victim(Pool& pool) const;
    void upload(const Pool& pool, std::uint32_t slot, const std::byte* staging, cudaStream_t stream) const;
    void stage(const Pool& pool, std::size_t layer, std::int32_t expert, std::byte* staging) const;
    [[nodiscard]] ops::cpu::HostExpertView host_view(std::size_t layer, std::int32_t expert) const;
    [[nodiscard]] static PlaneLayout plane_layout(artifact::NumericFormat down_format);

    std::array<HostRoutedBanks, kOffloadTextLayers> banks_;
    std::array<HostPlanes, kOffloadTextLayers> host_planes_{};
    ExpertOffloadPolicy policy_;
    std::size_t requested_bytes_ = 0;
    std::int32_t max_tokens_     = 1;
    bool allocated_pools_        = false;
    std::vector<Pool> pools_;
    std::vector<std::size_t> empty_slots_; // per pool: slots that never held an expert
    std::array<std::int32_t, kOffloadTextLayers> pool_of_layer_{};
    std::unordered_map<std::int64_t, std::uint32_t> resident_; // key -> slot, uploads complete
    std::uint64_t clock_ = 0;
    std::vector<std::uint8_t> needed_;
    std::vector<std::int32_t> distinct_;
    std::vector<std::int32_t> missing_;
    std::vector<std::uint16_t> miss_counts_; // [layer * 256 + expert]
    std::uint64_t decode_calls_ = 0;

    PinnedHostBuffer host_ids_;
    PinnedHostBuffer host_bank_of_expert_;
    PinnedHostBuffer host_alpha_;
    PinnedHostBuffer host_x_;
    PinnedHostBuffer host_cold_sum_;
    // Foreground uploads: two buffers alternating on the caller's stream.
    std::array<std::unique_ptr<PinnedHostBuffer>, 2> staging_;
    std::array<cudaEvent_t, 2> staging_free_{};
    std::array<bool, 2> staging_pending_{};
    // Background promotions: a ring of buffers, each owned by one in-flight upload.
    std::vector<std::unique_ptr<PinnedHostBuffer>> promotion_buffers_;
    std::vector<cudaEvent_t> promotion_done_;
    std::vector<bool> promotion_busy_;
    std::vector<Promotion> promotions_;
    cudaStream_t copy_stream_  = nullptr;
    cudaEvent_t compute_mark_  = nullptr;

    // Prefill prefetch: jobs planned on the calling thread, staged and uploaded by prefetch_thread_.
    std::unordered_map<std::int64_t, std::unique_ptr<PrefetchJob>> prefetches_; // key -> job
    std::deque<PrefetchJob*> prefetch_queue_;
    std::mutex prefetch_mutex_;
    std::condition_variable prefetch_wake_;
    std::condition_variable prefetch_issued_;
    bool prefetch_stop_ = false;
    std::vector<cudaEvent_t> free_events_;
    std::array<std::unique_ptr<PinnedHostBuffer>, 4> prefetch_buffers_;
    std::array<cudaEvent_t, 4> prefetch_buffer_free_{};
    std::thread prefetch_thread_;

    std::unique_ptr<HostWorkerPool> workers_;
    std::unique_ptr<ops::cpu::SpinPool> cpu_pool_;
    std::FILE* trace_ = nullptr;
    ExpertOffloadStats stats_;
};

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail
