#pragma once

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/host_worker_pool.h"
#include "core/tensor.h"
#include "ninfer/ops/sparse_moe.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
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

struct ExpertOffloadStats {
    std::uint64_t calls        = 0;
    std::uint64_t lookups      = 0;
    std::uint64_t misses       = 0;
    std::uint64_t bytes_copied = 0;
    double stage_seconds       = 0.0;
};

// Routed experts of the 35B-A3B Text layers held in host memory (the artifact file mapping, so
// normally the page cache) with a device LRU cache of expert slots. Layers whose routed down bank
// has the same format share one slot pool; a pool is a pair of row-split banks with `slots`
// experts in the registered per-expert row geometry, which SparseMoe reads through its
// SparseMoeExpertResidency. Experts reach a slot through two pinned staging buffers, so the host
// copy of one expert overlaps the upload of the previous one; all device work is ordered on the
// caller's stream, so an evicted slot was last read by earlier work on that stream.
class ExpertOffload {
public:
    ExpertOffload(std::array<HostRoutedBanks, kOffloadTextLayers> banks, std::size_t requested_bytes,
                  std::int32_t max_tokens);
    ~ExpertOffload();

    ExpertOffload(const ExpertOffload&)            = delete;
    ExpertOffload& operator=(const ExpertOffload&) = delete;

    // Minimum device bytes: 256 slots (one full layer, which a long prefill chunk can touch) per
    // down format.
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
    };
    // Byte geometry of one expert inside host banks and device pools.
    struct PlaneLayout {
        std::size_t gate_code_row = 0, gate_scale_row = 0;
        std::size_t down_code_row = 0, down_high_row = 0, down_scale_row = 0;
        [[nodiscard]] std::size_t expert_bytes() const noexcept;
    };
    struct Pool {
        artifact::NumericFormat down_format = artifact::NumericFormat::Q5G64_F16S;
        std::uint32_t layers                = 0;
        PlaneLayout planes;
        std::vector<Slot> slots;
        DeviceBuffer gate_codes, gate_scales, down_codes, down_high, down_scales;
        Weight gate_up, down;
    };

    void acquire(std::size_t layer, std::span<const std::int32_t> selected, std::int32_t* bank_of_expert,
                 cudaStream_t stream);
    void stage(const Pool& pool, std::size_t layer, std::int32_t expert, std::byte* staging) const;
    [[nodiscard]] static PlaneLayout plane_layout(artifact::NumericFormat down_format);

    std::array<HostRoutedBanks, kOffloadTextLayers> banks_;
    std::size_t requested_bytes_ = 0;
    bool allocated_pools_        = false;
    std::vector<Pool> pools_;
    std::array<std::int32_t, kOffloadTextLayers> pool_of_layer_{};
    std::unordered_map<std::int64_t, std::uint32_t> resident_; // key -> slot in its pool
    std::uint64_t clock_ = 0;
    std::vector<std::uint8_t> needed_;
    std::vector<std::int32_t> distinct_;
    PinnedHostBuffer host_ids_;
    PinnedHostBuffer host_bank_of_expert_;
    std::array<std::unique_ptr<PinnedHostBuffer>, 2> staging_;
    std::array<cudaEvent_t, 2> staging_free_{};
    std::array<bool, 2> staging_pending_{};
    std::unique_ptr<HostWorkerPool> workers_;
    ExpertOffloadStats stats_;
};

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail
