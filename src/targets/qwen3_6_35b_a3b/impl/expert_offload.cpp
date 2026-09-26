#include "targets/qwen3_6_35b_a3b/impl/expert_offload.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <future>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6_35b_a3b::detail {
namespace {

using artifact::NumericFormat;

constexpr std::int32_t kHidden       = 2048;
constexpr std::int32_t kIntermediate = 512;
constexpr std::int32_t kGateRows     = 2 * kIntermediate; // rows of one expert's gate/up
constexpr std::int32_t kDownRows     = kHidden;           // rows of one expert's down
// Every SparseMoe call may select all 256 experts of a layer.
constexpr std::uint32_t kMinimumSlots = kOffloadExperts;
// Largest bank the Op admits (32-bit row indices over 1 MiB of gate/up codes per expert).
constexpr std::uint32_t kMaximumSlots = 2047;
constexpr std::size_t kCopyChunkBytes = 512U << 10;
constexpr std::uint32_t kHostCopyThreads = 4;

std::array<std::uint64_t, 2> shape(std::int64_t rows, std::int64_t columns) {
    return {static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(columns)};
}

Weight row_split_view(QType qtype, std::int32_t group, std::int32_t rows, std::int32_t columns,
                      const DeviceBuffer& codes, const DeviceBuffer* high, const DeviceBuffer& scales) {
    Weight out{};
    out.payload          = codes.p;
    out.payload_bytes    = codes.bytes + (high != nullptr ? high->bytes : 0) + scales.bytes;
    out.high_plane_bytes = high != nullptr ? high->bytes : 0;
    out.qtype            = qtype;
    out.group_size       = static_cast<std::uint32_t>(group);
    out.group            = group;
    out.qdata            = codes.p;
    out.qhigh            = high != nullptr ? high->p : nullptr;
    out.scales           = scales.p;
    out.n                = rows;
    out.k                = columns;
    out.layout           = QuantLayout::RowSplit;
    out.scale_dtype      = DType::FP16;
    out.ndim             = 2;
    out.shape[0]         = rows;
    out.shape[1]         = columns;
    out.padded_shape[0]  = rows;
    out.padded_shape[1]  = columns;
    return out;
}

QType qtype_of(NumericFormat format) {
    switch (format) {
    case NumericFormat::Q5G64_F16S:
        return QType::Q5G64_F16S;
    case NumericFormat::Q6G64_F16S:
        return QType::Q6G64_F16S;
    default:
        throw std::invalid_argument("35B expert offload: routed down must be Q5 or Q6");
    }
}

} // namespace

std::size_t ExpertOffload::PlaneLayout::expert_bytes() const noexcept {
    return kGateRows * (gate_code_row + gate_scale_row) +
           kDownRows * (down_code_row + down_high_row + down_scale_row);
}

ExpertOffload::PlaneLayout ExpertOffload::plane_layout(NumericFormat down_format) {
    const auto gate = artifact::row_split_geometry(NumericFormat::Q4G64_F16S, shape(1, kHidden));
    const auto down = artifact::row_split_geometry(down_format, shape(1, kIntermediate));
    PlaneLayout out;
    out.gate_code_row  = gate.groups_per_row * gate.low_bytes_per_group;
    out.gate_scale_row = gate.groups_per_row * 2;
    out.down_code_row  = down.groups_per_row * down.low_bytes_per_group;
    out.down_high_row  = down.groups_per_row * down.high_bytes_per_group;
    out.down_scale_row = down.groups_per_row * 2;
    return out;
}

ExpertOffload::ExpertOffload(std::array<HostRoutedBanks, kOffloadTextLayers> banks,
                             std::size_t requested_bytes, std::int32_t max_tokens)
    : banks_(banks), requested_bytes_(requested_bytes),
      host_ids_(sizeof(std::int32_t) * 8 * static_cast<std::size_t>(std::max(max_tokens, 1))),
      host_bank_of_expert_(sizeof(std::int32_t) * kOffloadExperts),
      workers_(std::make_unique<HostWorkerPool>(kHostCopyThreads, 4 * kHostCopyThreads + 16)) {
    for (std::size_t layer = 0; layer < kOffloadTextLayers; ++layer) {
        const HostRoutedBanks& bank = banks_[layer];
        const auto gate = artifact::row_split_geometry(NumericFormat::Q4G64_F16S,
                                                       shape(kOffloadExperts * kGateRows, kHidden));
        const auto down = artifact::row_split_geometry(bank.down_format,
                                                       shape(kOffloadExperts * kDownRows, kIntermediate));
        if (bank.gate_up.size() != gate.encoded_bytes || bank.down.size() != down.encoded_bytes) {
            throw std::invalid_argument("35B expert offload: routed bank payload has the wrong size");
        }
        auto found = std::find_if(pools_.begin(), pools_.end(),
                                  [&](const Pool& pool) { return pool.down_format == bank.down_format; });
        if (found == pools_.end()) {
            Pool pool;
            pool.down_format = bank.down_format;
            pool.planes      = plane_layout(bank.down_format);
            pools_.push_back(std::move(pool));
            found = pools_.end() - 1;
        }
        ++found->layers;
        pool_of_layer_[layer] = static_cast<std::int32_t>(found - pools_.begin());
    }
    std::size_t staging_bytes = 0;
    for (const Pool& pool : pools_) { staging_bytes = std::max(staging_bytes, pool.planes.expert_bytes()); }
    for (auto& buffer : staging_) { buffer = std::make_unique<PinnedHostBuffer>(staging_bytes); }
    for (auto& event : staging_free_) { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
}

ExpertOffload::~ExpertOffload() {
    for (auto& event : staging_free_) {
        if (event != nullptr) {
            (void)cudaEventSynchronize(event);
            (void)cudaEventDestroy(event);
        }
    }
    if (stats_.calls != 0) {
        const double hit = stats_.lookups == 0 ? 0.0
                                               : 100.0 * static_cast<double>(stats_.lookups - stats_.misses) /
                                                     static_cast<double>(stats_.lookups);
        std::fprintf(stderr,
                     "[qwen3.6-35b-a3b expert offload] %llu MoE calls, %llu expert lookups, %.1f%% cache "
                     "hits, %.2f GiB copied, %.2f s staging\n",
                     static_cast<unsigned long long>(stats_.calls),
                     static_cast<unsigned long long>(stats_.lookups), hit,
                     static_cast<double>(stats_.bytes_copied) / static_cast<double>(1ULL << 30),
                     stats_.stage_seconds);
    }
}

std::size_t ExpertOffload::minimum_device_bytes() const noexcept {
    std::size_t total = 0;
    for (const Pool& pool : pools_) { total += kMinimumSlots * pool.planes.expert_bytes(); }
    return total;
}

void ExpertOffload::allocate(std::size_t budget_bytes) {
    if (allocated_pools_) { throw std::logic_error("35B expert offload is already allocated"); }
    const std::size_t minimum = minimum_device_bytes();
    if (budget_bytes < minimum) {
        throw std::invalid_argument(
            "expert offload needs at least " + std::to_string((minimum + (1U << 20) - 1) >> 20) +
            " MiB of device memory for its expert cache (" + std::to_string(budget_bytes >> 20) +
            " MiB available); lower --max-context or --prefill-chunk, or pass --expert-cache-gib");
    }
    // Every pool keeps its minimum; the rest gives each pool the same resident fraction of the
    // experts its minimum leaves uncovered.
    double uncovered = 0.0;
    for (const Pool& pool : pools_) {
        const std::uint32_t capacity = std::min(pool.layers * kOffloadExperts, kMaximumSlots);
        uncovered += static_cast<double>(capacity - kMinimumSlots) * static_cast<double>(pool.planes.expert_bytes());
    }
    const double fraction =
        uncovered > 0.0 ? std::min(1.0, static_cast<double>(budget_bytes - minimum) / uncovered) : 0.0;
    std::size_t total_slots = 0, total_bytes = 0;
    for (Pool& pool : pools_) {
        const std::uint32_t capacity = std::min(pool.layers * kOffloadExperts, kMaximumSlots);
        const auto slots = kMinimumSlots + static_cast<std::uint32_t>(fraction * (capacity - kMinimumSlots));
        const PlaneLayout& p = pool.planes;
        pool.gate_codes  = DeviceBuffer(static_cast<std::size_t>(slots) * kGateRows * p.gate_code_row);
        pool.gate_scales = DeviceBuffer(static_cast<std::size_t>(slots) * kGateRows * p.gate_scale_row);
        pool.down_codes  = DeviceBuffer(static_cast<std::size_t>(slots) * kDownRows * p.down_code_row);
        pool.down_high   = DeviceBuffer(static_cast<std::size_t>(slots) * kDownRows * p.down_high_row);
        pool.down_scales = DeviceBuffer(static_cast<std::size_t>(slots) * kDownRows * p.down_scale_row);
        pool.gate_up = row_split_view(QType::Q4G64_F16S, 64, static_cast<std::int32_t>(slots) * kGateRows, kHidden,
                                      pool.gate_codes, nullptr, pool.gate_scales);
        pool.down    = row_split_view(qtype_of(pool.down_format), 64, static_cast<std::int32_t>(slots) * kDownRows,
                                      kIntermediate, pool.down_codes, &pool.down_high, pool.down_scales);
        pool.slots.assign(slots, Slot{});
        total_slots += slots;
        total_bytes += static_cast<std::size_t>(slots) * p.expert_bytes();
    }
    resident_.reserve(total_slots * 2);
    allocated_pools_ = true;
    std::fprintf(stderr,
                 "[qwen3.6-35b-a3b expert offload] device expert cache: %zu of %zu experts (%.2f GiB) in "
                 "%zu slot pool(s); routed experts stream from host memory\n",
                 total_slots, kOffloadTextLayers * static_cast<std::size_t>(kOffloadExperts),
                 static_cast<double>(total_bytes) / static_cast<double>(1ULL << 30), pools_.size());
}

void ExpertOffload::stage(const Pool& pool, std::size_t layer, std::int32_t expert, std::byte* staging) const {
    struct Copy {
        std::byte* destination;
        const std::byte* source;
        std::size_t bytes;
    };
    const PlaneLayout& p = pool.planes;
    const HostRoutedBanks& bank = banks_[layer];
    const auto gate = artifact::row_split_geometry(NumericFormat::Q4G64_F16S,
                                                   shape(kOffloadExperts * kGateRows, kHidden));
    const auto down = artifact::row_split_geometry(bank.down_format,
                                                   shape(kOffloadExperts * kDownRows, kIntermediate));
    const std::size_t e = static_cast<std::size_t>(expert);
    std::array<Copy, 5> copies{};
    std::byte* cursor = staging;
    const auto add = [&](std::size_t index, std::span<const std::byte> source, std::size_t plane_offset,
                         std::size_t rows, std::size_t row_bytes) {
        copies[index] = {cursor, source.data() + plane_offset + e * rows * row_bytes, rows * row_bytes};
        cursor += rows * row_bytes;
    };
    add(0, bank.gate_up, 0, kGateRows, p.gate_code_row);
    add(1, bank.gate_up, gate.scale_plane_offset, kGateRows, p.gate_scale_row);
    add(2, bank.down, 0, kDownRows, p.down_code_row);
    add(3, bank.down, down.high_plane_offset, kDownRows, p.down_high_row);
    add(4, bank.down, down.scale_plane_offset, kDownRows, p.down_scale_row);

    std::vector<std::future<void>> pending;
    pending.reserve(8);
    for (const Copy& copy : copies) {
        for (std::size_t offset = 0; offset < copy.bytes; offset += kCopyChunkBytes) {
            const std::size_t bytes = std::min(kCopyChunkBytes, copy.bytes - offset);
            pending.push_back(workers_->submit(
                [copy, offset, bytes] { std::memcpy(copy.destination + offset, copy.source + offset, bytes); }));
        }
    }
    for (auto& future : pending) { future.get(); }
}

void ExpertOffload::acquire(std::size_t layer, std::span<const std::int32_t> selected,
                            std::int32_t* bank_of_expert, cudaStream_t stream) {
    Pool& pool          = pools_[static_cast<std::size_t>(pool_of_layer_[layer])];
    std::vector<Slot>& slots = pool.slots;
    const std::uint64_t tick = ++clock_;
    needed_.assign(slots.size(), 0);
    distinct_.clear();
    std::array<bool, kOffloadExperts> seen{};
    for (const std::int32_t expert : selected) {
        if (expert < 0 || expert >= kOffloadExperts) {
            throw std::logic_error("35B expert offload: router produced an invalid expert id");
        }
        if (!seen[expert]) {
            seen[expert] = true;
            distinct_.push_back(expert);
        }
    }
    std::vector<std::int32_t> missing;
    for (const std::int32_t expert : distinct_) {
        const std::int64_t key = static_cast<std::int64_t>(layer) * kOffloadExperts + expert;
        ++stats_.lookups;
        const auto found = resident_.find(key);
        if (found != resident_.end()) {
            slots[found->second].last_used = tick;
            needed_[found->second]         = 1;
            bank_of_expert[expert]         = static_cast<std::int32_t>(found->second);
        } else {
            missing.push_back(expert);
        }
    }
    if (missing.empty()) { return; }

    std::vector<std::uint32_t> candidates;
    candidates.reserve(slots.size());
    for (std::uint32_t index = 0; index < slots.size(); ++index) {
        if (!needed_[index]) { candidates.push_back(index); }
    }
    if (candidates.size() < missing.size()) {
        throw std::logic_error("35B expert offload: request exceeds the evictable slots");
    }
    std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(missing.size()),
                      candidates.end(), [&](std::uint32_t a, std::uint32_t b) {
                          const bool a_empty = slots[a].key < 0;
                          const bool b_empty = slots[b].key < 0;
                          if (a_empty != b_empty) { return a_empty; }
                          return slots[a].last_used < slots[b].last_used;
                      });

    const auto started   = std::chrono::steady_clock::now();
    const PlaneLayout& p = pool.planes;
    for (std::size_t m = 0; m < missing.size(); ++m) {
        const std::int32_t expert  = missing[m];
        const std::uint32_t victim = candidates[m];
        Slot& slot                 = slots[victim];
        if (slot.key >= 0) { resident_.erase(slot.key); }

        const std::size_t buffer = m % staging_.size();
        if (staging_pending_[buffer]) { CUDA_CHECK(cudaEventSynchronize(staging_free_[buffer])); }
        auto* staging = static_cast<std::byte*>(staging_[buffer]->data());
        stage(pool, layer, expert, staging);

        const std::size_t v = victim;
        const std::byte* cursor = staging;
        const auto upload = [&](const DeviceBuffer& plane, std::size_t rows, std::size_t row_bytes) {
            const std::size_t bytes = rows * row_bytes;
            if (bytes == 0) { return; }
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(plane.p) + v * bytes, cursor, bytes,
                                       cudaMemcpyHostToDevice, stream));
            cursor += bytes;
        };
        upload(pool.gate_codes, kGateRows, p.gate_code_row);
        upload(pool.gate_scales, kGateRows, p.gate_scale_row);
        upload(pool.down_codes, kDownRows, p.down_code_row);
        upload(pool.down_high, kDownRows, p.down_high_row);
        upload(pool.down_scales, kDownRows, p.down_scale_row);
        CUDA_CHECK(cudaEventRecord(staging_free_[buffer], stream));
        staging_pending_[buffer] = true;

        slot.key       = static_cast<std::int64_t>(layer) * kOffloadExperts + expert;
        slot.last_used = tick;
        resident_[slot.key]    = victim;
        needed_[victim]        = 1;
        bank_of_expert[expert] = static_cast<std::int32_t>(victim);
        ++stats_.misses;
        stats_.bytes_copied += p.expert_bytes();
    }
    stats_.stage_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

void ExpertOffload::sparse_moe(std::size_t layer, const Tensor& x, const ops::SparseMoeWeights& resident,
                               Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    if (!allocated_pools_) { throw std::logic_error("35B expert offload was not allocated"); }
    if (layer >= kOffloadTextLayers) { throw std::out_of_range("35B expert offload: layer out of range"); }
    const Pool& pool = pools_[static_cast<std::size_t>(pool_of_layer_[layer])];
    if (static_cast<std::size_t>(x.ne[1]) * 8 * sizeof(std::int32_t) > host_ids_.size()) {
        throw std::invalid_argument("35B expert offload: token count exceeds the planned maximum");
    }
    ops::SparseMoeWeights weights = resident;
    weights.routed_gate_up        = pool.gate_up;
    weights.routed_down           = pool.down;
    const ops::SparseMoeExpertResidency residency{
        .host_ids            = static_cast<std::int32_t*>(host_ids_.data()),
        .host_bank_of_expert = static_cast<std::int32_t*>(host_bank_of_expert_.data()),
        .banks               = static_cast<std::int32_t>(pool.slots.size()),
        .acquire             = [this, layer](std::span<const std::int32_t> selected, std::int32_t* bank_of_expert,
                                 cudaStream_t s) { acquire(layer, selected, bank_of_expert, s); },
    };
    ++stats_.calls;
    auto scope               = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(ops::sparse_moe_workspace_capacity_bytes(
        weights.routed_gate_up.qtype, weights.routed_down.qtype, x.ne[1], x.ne[1]));
    WorkspaceArena leaf_workspace(storage);
    ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::AddResidual, residual, residency, leaf_workspace,
                    stream);
}

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail
