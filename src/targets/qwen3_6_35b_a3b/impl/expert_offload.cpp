#include "targets/qwen3_6_35b_a3b/impl/expert_offload.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>

#include <sys/mman.h>
#include <unistd.h>

namespace ninfer::targets::qwen3_6_35b_a3b::detail {
namespace {

using artifact::NumericFormat;

constexpr std::int32_t kHidden       = 2048;
constexpr std::int32_t kIntermediate = 512;
constexpr std::int32_t kTopK         = 8;
constexpr std::int32_t kGateRows     = 2 * kIntermediate; // rows of one expert's gate/up
constexpr std::int32_t kDownRows     = kHidden;           // rows of one expert's down
// Every prefill-size SparseMoe call may select all 256 experts of a layer.
constexpr std::uint32_t kMinimumSlots = kOffloadExperts;
// Largest bank the Op admits (32-bit row indices over 1 MiB of gate/up codes per expert), less
// the zero bank.
constexpr std::uint32_t kMaximumSlots = 2046;
constexpr std::size_t kCopyChunkBytes = 512U << 10;
constexpr std::uint32_t kHostCopyThreads = 4;
constexpr std::size_t kPromotionBuffers  = 8;

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

std::uint32_t env_u32(const char* name, std::uint32_t fallback) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' ? static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10))
                                                : fallback;
}

// Populates the page tables of the routed banks. Reading through a fresh file mapping takes a
// minor fault per 4 KiB page, which cuts host expert throughput to about a quarter (measured
// 4.6 vs 17.3 GB/s); prefaulting at load pays that once, in parallel.
void prefault(const std::vector<std::span<const std::byte>>& ranges, unsigned threads) {
    const std::size_t page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    constexpr std::size_t kChunk = 64ULL << 20;
    std::vector<std::pair<const std::byte*, std::size_t>> chunks;
    for (const auto& range : ranges) {
        const auto begin = reinterpret_cast<std::uintptr_t>(range.data()) / page * page;
        const auto end   = reinterpret_cast<std::uintptr_t>(range.data() + range.size());
        for (std::uintptr_t at = begin; at < end; at += kChunk) {
            chunks.emplace_back(reinterpret_cast<const std::byte*>(at), std::min<std::uintptr_t>(kChunk, end - at));
        }
    }
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t) {
        pool.emplace_back([&] {
            for (std::size_t i = next++; i < chunks.size(); i = next++) {
                auto* start = const_cast<std::byte*>(chunks[i].first);
#ifdef MADV_POPULATE_READ
                if (madvise(start, chunks[i].second, MADV_POPULATE_READ) == 0) { continue; }
#endif
                volatile std::uint8_t sink = 0;
                for (std::size_t off = 0; off < chunks[i].second; off += page) {
                    sink = sink + static_cast<std::uint8_t>(start[off]);
                }
            }
        });
    }
    for (auto& thread : pool) { thread.join(); }
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

} // namespace

ExpertOffloadPolicy ExpertOffloadPolicy::from_environment() {
    ExpertOffloadPolicy out;
    out.host_cold_experts   = env_u32("NINFER_OFFLOAD_COLD", out.host_cold_experts ? 1 : 0) != 0;
    out.promotions_per_call = env_u32("NINFER_OFFLOAD_PROMOTE", out.promotions_per_call);
    out.host_threads        = env_u32("NINFER_OFFLOAD_CPU_THREADS", out.host_threads);
    if (out.host_threads == 0) { out.host_threads = std::max(1U, std::thread::hardware_concurrency() / 2); }
    out.admit_misses        = std::max(1U, env_u32("NINFER_OFFLOAD_ADMIT", out.admit_misses));
    out.prefetch_min_tokens = env_u32("NINFER_OFFLOAD_PREFETCH", out.prefetch_min_tokens);
    return out;
}

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

ExpertOffload::ExpertOffload(std::array<HostRoutedBanks, kOffloadTextLayers> banks, std::size_t requested_bytes,
                             std::int32_t max_tokens, ExpertOffloadPolicy policy)
    : banks_(banks), policy_(policy), requested_bytes_(requested_bytes), max_tokens_(std::max(max_tokens, 1)),
      host_ids_(sizeof(std::int32_t) * 2 * kTopK * static_cast<std::size_t>(std::max(max_tokens, 1))),
      host_bank_of_expert_(sizeof(std::int32_t) * kOffloadExperts),
      host_alpha_(sizeof(float) * kTopK * static_cast<std::size_t>(std::max(max_tokens, 1))),
      host_x_(sizeof(std::uint16_t) * kHidden * static_cast<std::size_t>(std::max(max_tokens, 1))),
      host_cold_sum_(sizeof(float) * kHidden * static_cast<std::size_t>(std::max(max_tokens, 1))),
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
        host_planes_[layer] = HostPlanes{gate.scale_plane_offset, down.high_plane_offset, down.scale_plane_offset};
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
    for (std::size_t i = 0; i < kPromotionBuffers; ++i) {
        promotion_buffers_.push_back(std::make_unique<PinnedHostBuffer>(staging_bytes));
        cudaEvent_t event = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        promotion_done_.push_back(event);
        promotion_busy_.push_back(false);
    }
    for (std::size_t i = 0; i < prefetch_buffers_.size(); ++i) {
        prefetch_buffers_[i] = std::make_unique<PinnedHostBuffer>(staging_bytes);
        CUDA_CHECK(cudaEventCreateWithFlags(&prefetch_buffer_free_[i], cudaEventDisableTiming));
    }
    int least = 0, greatest = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&copy_stream_, cudaStreamNonBlocking, least));
    CUDA_CHECK(cudaEventCreateWithFlags(&compute_mark_, cudaEventDisableTiming));
    cpu_pool_ = std::make_unique<ops::cpu::SpinPool>(policy_.host_threads);
    miss_counts_.assign(kOffloadTextLayers * kOffloadExperts, 0);
    {
        const auto started = std::chrono::steady_clock::now();
        std::vector<std::span<const std::byte>> ranges;
        for (const HostRoutedBanks& bank : banks_) {
            ranges.push_back(bank.gate_up);
            ranges.push_back(bank.down);
        }
        prefault(ranges, 8);
        std::fprintf(stderr, "[qwen3.6-35b-a3b expert offload] prefaulted routed experts in %.1f s\n",
                     seconds_since(started));
    }
    if (const char* path = std::getenv("NINFER_OFFLOAD_TRACE"); path != nullptr && path[0] != '\0') {
        trace_ = std::fopen(path, "w");
    }
    prefetch_thread_ = std::thread([this] { prefetch_loop(); });
}

ExpertOffload::~ExpertOffload() {
    {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        prefetch_stop_ = true;
    }
    prefetch_wake_.notify_all();
    if (prefetch_thread_.joinable()) { prefetch_thread_.join(); }
    if (copy_stream_ != nullptr) { (void)cudaStreamSynchronize(copy_stream_); }
    for (auto& [key, job] : prefetches_) { (void)cudaEventDestroy(job->done); }
    for (cudaEvent_t event : free_events_) { (void)cudaEventDestroy(event); }
    for (cudaEvent_t event : prefetch_buffer_free_) {
        if (event != nullptr) { (void)cudaEventDestroy(event); }
    }
    for (auto& event : staging_free_) {
        if (event != nullptr) {
            (void)cudaEventSynchronize(event);
            (void)cudaEventDestroy(event);
        }
    }
    for (auto& event : promotion_done_) { (void)cudaEventDestroy(event); }
    if (compute_mark_ != nullptr) { (void)cudaEventDestroy(compute_mark_); }
    if (copy_stream_ != nullptr) { (void)cudaStreamDestroy(copy_stream_); }
    if (trace_ != nullptr) { std::fclose(trace_); }
    if (stats_.calls != 0) {
        const auto pct = [&](std::uint64_t part) {
            return stats_.lookups == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(stats_.lookups);
        };
        std::fprintf(stderr,
                     "[qwen3.6-35b-a3b expert offload] %llu MoE calls, %llu expert lookups, %.1f%% cache hits, "
                     "%.1f%% host-computed, %.1f%% uploaded before use, %llu background promotions, %llu of %llu "
                     "prefetched claimed in flight, %.2f GiB copied, %.2f s staging, %.2f s host experts\n",
                     static_cast<unsigned long long>(stats_.calls), static_cast<unsigned long long>(stats_.lookups),
                     pct(stats_.hits), pct(stats_.cold), pct(stats_.uploads),
                     static_cast<unsigned long long>(stats_.promotions),
                     static_cast<unsigned long long>(stats_.prefetch_used),
                     static_cast<unsigned long long>(stats_.prefetched),
                     static_cast<double>(stats_.bytes_copied) / static_cast<double>(1ULL << 30),
                     stats_.upload_seconds, stats_.host_compute_seconds);
        if (stats_.prefill_moe_seconds > 0) {
            std::fprintf(stderr, "[qwen3.6-35b-a3b expert offload] prefill MoE wall time %.2f s\n",
                         stats_.prefill_moe_seconds);
        }
    }
}

std::size_t ExpertOffload::minimum_device_bytes() const noexcept {
    std::size_t total = 0;
    for (const Pool& pool : pools_) { total += (kMinimumSlots + 1) * pool.planes.expert_bytes(); }
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
        const std::size_t banks = static_cast<std::size_t>(slots) + 1; // + the zero bank
        const PlaneLayout& p    = pool.planes;
        pool.gate_codes  = DeviceBuffer(banks * kGateRows * p.gate_code_row);
        pool.gate_scales = DeviceBuffer(banks * kGateRows * p.gate_scale_row);
        pool.down_codes  = DeviceBuffer(banks * kDownRows * p.down_code_row);
        pool.down_high   = DeviceBuffer(banks * kDownRows * p.down_high_row);
        pool.down_scales = DeviceBuffer(banks * kDownRows * p.down_scale_row);
        // The zero bank: zero codes with zero scales decode to exactly zero weights.
        for (auto [plane, rows, row_bytes] :
             {std::tuple{&pool.gate_codes, kGateRows, p.gate_code_row},
              std::tuple{&pool.gate_scales, kGateRows, p.gate_scale_row},
              std::tuple{&pool.down_codes, kDownRows, p.down_code_row},
              std::tuple{&pool.down_high, kDownRows, p.down_high_row},
              std::tuple{&pool.down_scales, kDownRows, p.down_scale_row}}) {
            const std::size_t bytes = static_cast<std::size_t>(rows) * row_bytes;
            if (bytes != 0) {
                CUDA_CHECK(cudaMemset(static_cast<std::byte*>(plane->p) + slots * bytes, 0, bytes));
            }
        }
        pool.gate_up = row_split_view(QType::Q4G64_F16S, 64, static_cast<std::int32_t>(banks) * kGateRows, kHidden,
                                      pool.gate_codes, nullptr, pool.gate_scales);
        pool.down    = row_split_view(qtype_of(pool.down_format), 64, static_cast<std::int32_t>(banks) * kDownRows,
                                      kIntermediate, pool.down_codes, &pool.down_high, pool.down_scales);
        pool.slots.assign(slots, Slot{});
        empty_slots_.push_back(slots);
        total_slots += slots;
        total_bytes += banks * p.expert_bytes();
    }
    resident_.reserve(total_slots * 2);
    allocated_pools_ = true;
    std::fprintf(stderr,
                 "[qwen3.6-35b-a3b expert offload] device expert cache: %zu of %zu experts (%.2f GiB) in "
                 "%zu slot pool(s); host experts %s, %u background promotions per call, %u host threads\n",
                 total_slots, kOffloadTextLayers * static_cast<std::size_t>(kOffloadExperts),
                 static_cast<double>(total_bytes) / static_cast<double>(1ULL << 30), pools_.size(),
                 policy_.host_cold_experts ? "on" : "off", policy_.promotions_per_call, cpu_pool_->threads());
}

ops::cpu::HostExpertView ExpertOffload::host_view(std::size_t layer, std::int32_t expert) const {
    const Pool& pool            = pools_[static_cast<std::size_t>(pool_of_layer_[layer])];
    const PlaneLayout& p        = pool.planes;
    const HostRoutedBanks& bank = banks_[layer];
    const HostPlanes& planes    = host_planes_[layer];
    const auto* gate            = reinterpret_cast<const std::uint8_t*>(bank.gate_up.data());
    const auto* down            = reinterpret_cast<const std::uint8_t*>(bank.down.data());
    const std::size_t e         = static_cast<std::size_t>(expert);
    return ops::cpu::HostExpertView{
        .gate_codes  = gate + e * kGateRows * p.gate_code_row,
        .gate_scales = gate + planes.gate_scale_offset + e * kGateRows * p.gate_scale_row,
        .down_codes  = down + e * kDownRows * p.down_code_row,
        .down_high   = down + planes.down_high_offset + e * kDownRows * p.down_high_row,
        .down_scales = down + planes.down_scale_offset + e * kDownRows * p.down_scale_row,
        .down_qtype  = qtype_of(pool.down_format),
    };
}

void ExpertOffload::stage(const Pool& pool, std::size_t layer, std::int32_t expert, std::byte* staging) const {
    struct Copy {
        std::byte* destination;
        const std::uint8_t* source;
        std::size_t bytes;
    };
    const PlaneLayout& p                 = pool.planes;
    const ops::cpu::HostExpertView view = host_view(layer, expert);
    std::array<Copy, 5> copies{};
    std::byte* cursor = staging;
    std::size_t count = 0;
    for (auto [source, bytes] : {std::pair{view.gate_codes, kGateRows * p.gate_code_row},
                                 std::pair{view.gate_scales, kGateRows * p.gate_scale_row},
                                 std::pair{view.down_codes, kDownRows * p.down_code_row},
                                 std::pair{view.down_high, kDownRows * p.down_high_row},
                                 std::pair{view.down_scales, kDownRows * p.down_scale_row}}) {
        copies[count++] = {cursor, source, bytes};
        cursor += bytes;
    }
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

void ExpertOffload::upload(const Pool& pool, std::uint32_t slot, const std::byte* staging, cudaStream_t stream) const {
    const PlaneLayout& p    = pool.planes;
    const std::size_t s     = slot;
    const std::byte* cursor = staging;
    for (auto [plane, rows, row_bytes] : {std::tuple{&pool.gate_codes, kGateRows, p.gate_code_row},
                                          std::tuple{&pool.gate_scales, kGateRows, p.gate_scale_row},
                                          std::tuple{&pool.down_codes, kDownRows, p.down_code_row},
                                          std::tuple{&pool.down_high, kDownRows, p.down_high_row},
                                          std::tuple{&pool.down_scales, kDownRows, p.down_scale_row}}) {
        const std::size_t bytes = static_cast<std::size_t>(rows) * row_bytes;
        if (bytes == 0) { continue; }
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(plane->p) + s * bytes, cursor, bytes,
                                   cudaMemcpyHostToDevice, stream));
        cursor += bytes;
    }
}

void ExpertOffload::retire_promotions() {
    for (std::size_t i = 0; i < promotions_.size();) {
        const Promotion& promotion = promotions_[i];
        const cudaError_t status   = cudaEventQuery(promotion_done_[promotion.buffer]);
        if (status == cudaErrorNotReady) {
            ++i;
            continue;
        }
        CUDA_CHECK(status);
        Slot& slot    = pools_[promotion.pool].slots[promotion.slot];
        slot.pending  = false;
        resident_[slot.key]                 = promotion.slot;
        promotion_busy_[promotion.buffer]   = false;
        promotions_[i]                      = promotions_.back();
        promotions_.pop_back();
    }
}

void ExpertOffload::prefetch_loop() {
    std::size_t turn = 0;
    for (;;) {
        PrefetchJob* job = nullptr;
        {
            std::unique_lock<std::mutex> lock(prefetch_mutex_);
            prefetch_wake_.wait(lock, [&] { return prefetch_stop_ || !prefetch_queue_.empty(); });
            if (prefetch_queue_.empty()) { return; }
            job = prefetch_queue_.front();
            prefetch_queue_.pop_front();
        }
        const std::size_t b = turn++ % prefetch_buffers_.size();
        CUDA_CHECK(cudaEventSynchronize(prefetch_buffer_free_[b]));
        auto* staging    = static_cast<std::byte*>(prefetch_buffers_[b]->data());
        const Pool& pool = pools_[job->pool];
        stage(pool, job->layer, job->expert, staging);
        upload(pool, job->slot, staging, copy_stream_);
        CUDA_CHECK(cudaEventRecord(job->done, copy_stream_));
        CUDA_CHECK(cudaEventRecord(prefetch_buffer_free_[b], copy_stream_));
        {
            std::lock_guard<std::mutex> lock(prefetch_mutex_);
            job->issued = true;
        }
        prefetch_issued_.notify_all();
    }
}

void ExpertOffload::plan_prefetch(std::size_t layer, std::size_t pool_index) {
    const std::size_t next = layer + 1;
    if (next >= kOffloadTextLayers) { return; }
    const std::size_t next_pool = static_cast<std::size_t>(pool_of_layer_[next]);
    Pool& pool                  = pools_[next_pool];
    // Slots the current layer reads and the next layer's resident experts must survive.
    std::vector<std::uint8_t> keep = next_pool == pool_index ? needed_ : std::vector<std::uint8_t>(pool.slots.size(), 0);
    for (std::uint32_t index = 0; index < pool.slots.size(); ++index) {
        const std::int64_t key = pool.slots[index].key;
        if (key >= 0 && static_cast<std::size_t>(key / kOffloadExperts) == next) { keep[index] = 1; }
    }
    std::vector<PrefetchJob*> planned;
    for (std::int32_t expert = 0; expert < kOffloadExperts; ++expert) {
        const std::int64_t key = static_cast<std::int64_t>(next) * kOffloadExperts + expert;
        if (resident_.count(key) != 0 || prefetches_.count(key) != 0) { continue; }
        std::int64_t best = -1;
        for (std::uint32_t index = 0; index < pool.slots.size(); ++index) {
            const Slot& slot = pool.slots[index];
            if (keep[index] || slot.pending) { continue; }
            if (slot.key < 0) {
                best = index;
                break;
            }
            if (best < 0 || slot.last_used < pool.slots[static_cast<std::size_t>(best)].last_used) { best = index; }
        }
        if (best < 0) { break; }
        Slot& target = pool.slots[static_cast<std::size_t>(best)];
        if (target.key >= 0) {
            resident_.erase(target.key);
        } else {
            --empty_slots_[next_pool];
        }
        target.key       = key;
        target.pending   = true;
        target.last_used = clock_;
        keep[static_cast<std::size_t>(best)] = 1;
        auto job    = std::make_unique<PrefetchJob>();
        job->pool   = next_pool;
        job->slot   = static_cast<std::uint32_t>(best);
        job->layer  = next;
        job->expert = expert;
        if (free_events_.empty()) {
            cudaEvent_t event = nullptr;
            CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            free_events_.push_back(event);
        }
        job->done = free_events_.back();
        free_events_.pop_back();
        planned.push_back(job.get());
        prefetches_[key] = std::move(job);
        ++stats_.prefetched;
        stats_.bytes_copied += pool.planes.expert_bytes();
    }
    if (planned.empty()) { return; }
    {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        for (PrefetchJob* job : planned) { prefetch_queue_.push_back(job); }
    }
    prefetch_wake_.notify_one();
}

void ExpertOffload::claim_prefetch(std::int64_t key, cudaStream_t stream) {
    auto found       = prefetches_.find(key);
    PrefetchJob* job = found->second.get();
    {
        std::unique_lock<std::mutex> lock(prefetch_mutex_);
        prefetch_issued_.wait(lock, [&] { return job->issued; });
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, job->done, 0));
    Slot& slot   = pools_[job->pool].slots[job->slot];
    slot.pending = false;
    resident_[key] = job->slot;
    free_events_.push_back(job->done); // later records do not affect the wait just enqueued
    prefetches_.erase(found);
    ++stats_.prefetch_used;
}

void ExpertOffload::retire_prefetches() {
    for (auto it = prefetches_.begin(); it != prefetches_.end();) {
        PrefetchJob* job = it->second.get();
        bool issued      = false;
        {
            std::lock_guard<std::mutex> lock(prefetch_mutex_);
            issued = job->issued;
        }
        if (!issued || cudaEventQuery(job->done) != cudaSuccess) {
            ++it;
            continue;
        }
        pools_[job->pool].slots[job->slot].pending = false;
        resident_[it->first]                        = job->slot;
        free_events_.push_back(job->done);
        it = prefetches_.erase(it);
    }
}

std::int64_t ExpertOffload::victim(Pool& pool) const {
    std::int64_t best = -1;
    for (std::uint32_t index = 0; index < pool.slots.size(); ++index) {
        const Slot& slot = pool.slots[index];
        if (needed_[index] || slot.pending) { continue; }
        if (slot.key < 0) { return index; }
        if (best < 0 || slot.last_used < pool.slots[static_cast<std::size_t>(best)].last_used) { best = index; }
    }
    return best;
}

void ExpertOffload::acquire(std::size_t layer, std::span<const std::int32_t> selected, std::int32_t* bank_of_expert,
                            bool cold_allowed, cudaStream_t stream) {
    retire_promotions();
    retire_prefetches();
    const std::size_t pool_index = static_cast<std::size_t>(pool_of_layer_[layer]);
    Pool& pool                   = pools_[pool_index];
    std::vector<Slot>& slots     = pool.slots;
    const std::uint64_t tick     = ++clock_;
    needed_.assign(slots.size(), 0);
    distinct_.clear();
    missing_.clear();
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
    if (trace_ != nullptr) {
        std::fprintf(trace_, "%zu,%zu", layer, selected.size() / kTopK);
        for (const std::int32_t expert : selected) { std::fprintf(trace_, ",%d", expert); }
        std::fputc('\n', trace_);
    }
    for (const std::int32_t expert : distinct_) {
        const std::int64_t key = static_cast<std::int64_t>(layer) * kOffloadExperts + expert;
        ++stats_.lookups;
        if (prefetches_.count(key) != 0) { claim_prefetch(key, stream); }
        const auto found = resident_.find(key);
        if (found != resident_.end()) {
            slots[found->second].last_used = tick;
            needed_[found->second]         = 1;
            bank_of_expert[expert]         = static_cast<std::int32_t>(found->second);
            ++stats_.hits;
        } else {
            missing_.push_back(expert);
        }
    }
    const std::size_t tokens  = selected.size() / kTopK;
    const bool prefetch_next = !cold_allowed && policy_.prefetch_min_tokens != 0 &&
                               tokens >= policy_.prefetch_min_tokens;
    if (missing_.empty()) {
        if (prefetch_next) { plan_prefetch(layer, pool_index); }
        return;
    }

    const auto started = std::chrono::steady_clock::now();
    if (cold_allowed && policy_.host_cold_experts) {
        // The host computes every missing expert for this call; a few are uploaded in the
        // background so the device cache follows the routing.
        std::uint32_t promoted = 0;
        bool marked            = false;
        // Decay the miss counts by half every 64 tokens (40 MoE calls per token).
        if (++decode_calls_ % (64 * kOffloadTextLayers) == 0) {
            for (auto& count : miss_counts_) { count = static_cast<std::uint16_t>(count >> 1); }
        }
        for (const std::int32_t expert : missing_) {
            bank_of_expert[expert] = -1;
            ++stats_.cold;
            const std::int64_t key = static_cast<std::int64_t>(layer) * kOffloadExperts + expert;
            std::uint16_t& misses  = miss_counts_[static_cast<std::size_t>(key)];
            if (misses < 0xFFFF) { ++misses; }
            // While the pool still has empty slots a promotion evicts nothing, so warm-up admits
            // every miss and uploads a few more per call.
            const bool warming = empty_slots_[pool_index] != 0;
            const std::uint32_t budget = warming ? std::max(policy_.promotions_per_call, 4U)
                                                 : policy_.promotions_per_call;
            if (promoted >= budget || (!warming && misses < policy_.admit_misses)) { continue; }
            bool in_flight         = prefetches_.count(key) != 0;
            for (const Promotion& promotion : promotions_) {
                in_flight = in_flight || (promotion.pool == pool_index && slots[promotion.slot].key == key);
            }
            if (in_flight) { continue; }
            const auto buffer = std::find(promotion_busy_.begin(), promotion_busy_.end(), false);
            const std::int64_t slot = victim(pool);
            if (buffer == promotion_busy_.end() || slot < 0) { break; }
            const std::size_t b = static_cast<std::size_t>(buffer - promotion_busy_.begin());
            if (!marked) {
                // Every earlier device read of the victim slot is on `stream` before this point.
                CUDA_CHECK(cudaEventRecord(compute_mark_, stream));
                CUDA_CHECK(cudaStreamWaitEvent(copy_stream_, compute_mark_, 0));
                marked = true;
            }
            Slot& target = slots[static_cast<std::size_t>(slot)];
            if (target.key >= 0) {
                resident_.erase(target.key);
            } else {
                --empty_slots_[pool_index];
            }
            target.key       = key;
            target.last_used = tick;
            target.pending   = true;
            stage(pool, layer, expert, static_cast<std::byte*>(promotion_buffers_[b]->data()));
            upload(pool, static_cast<std::uint32_t>(slot), static_cast<const std::byte*>(promotion_buffers_[b]->data()),
                   copy_stream_);
            CUDA_CHECK(cudaEventRecord(promotion_done_[b], copy_stream_));
            promotion_busy_[b] = true;
            promotions_.push_back(Promotion{pool_index, static_cast<std::uint32_t>(slot), b});
            ++promoted;
            ++stats_.promotions;
            stats_.bytes_copied += pool.planes.expert_bytes();
        }
        stats_.upload_seconds += seconds_since(started);
        return;
    }

    // Upload every missing expert before use, evicting least recently used unneeded slots.
    for (std::size_t m = 0; m < missing_.size(); ++m) {
        const std::int32_t expert = missing_[m];
        const std::int64_t slot   = victim(pool);
        if (slot < 0) { throw std::logic_error("35B expert offload: request exceeds the evictable slots"); }
        Slot& target = slots[static_cast<std::size_t>(slot)];
        if (target.key >= 0) {
            resident_.erase(target.key);
        } else {
            --empty_slots_[pool_index];
        }

        const std::size_t buffer = m % staging_.size();
        if (staging_pending_[buffer]) { CUDA_CHECK(cudaEventSynchronize(staging_free_[buffer])); }
        auto* staging = static_cast<std::byte*>(staging_[buffer]->data());
        stage(pool, layer, expert, staging);
        upload(pool, static_cast<std::uint32_t>(slot), staging, stream);
        CUDA_CHECK(cudaEventRecord(staging_free_[buffer], stream));
        staging_pending_[buffer] = true;

        target.key       = static_cast<std::int64_t>(layer) * kOffloadExperts + expert;
        target.last_used = tick;
        resident_[target.key]  = static_cast<std::uint32_t>(slot);
        needed_[static_cast<std::size_t>(slot)] = 1;
        bank_of_expert[expert] = static_cast<std::int32_t>(slot);
        ++stats_.uploads;
        stats_.bytes_copied += pool.planes.expert_bytes();
    }
    stats_.upload_seconds += seconds_since(started);
    if (prefetch_next) { plan_prefetch(layer, pool_index); }
}

void ExpertOffload::cold_compute(std::size_t layer, std::span<const std::int32_t> selected, const float* alpha,
                                 const std::uint16_t* x_bf16, std::int32_t tokens,
                                 const std::int32_t* bank_of_expert, float* cold_sum) {
    const auto started = std::chrono::steady_clock::now();
    std::vector<ops::cpu::ColdExpertWork> work;
    std::array<std::int32_t, kOffloadExperts> work_of_expert{};
    work_of_expert.fill(-1);
    for (std::size_t i = 0; i < selected.size(); ++i) {
        const std::int32_t expert = selected[i];
        if (bank_of_expert[expert] != -1) { continue; }
        if (work_of_expert[expert] < 0) {
            work_of_expert[expert] = static_cast<std::int32_t>(work.size());
            work.push_back({host_view(layer, expert), {}, {}});
        }
        auto& item = work[static_cast<std::size_t>(work_of_expert[expert])];
        item.columns.push_back(static_cast<std::int32_t>(i / kTopK));
        item.weights.push_back(alpha[i]);
    }
    ops::cpu::cold_experts(work, x_bf16, tokens, cold_sum, *cpu_pool_);
    stats_.host_compute_seconds += seconds_since(started);
}

void ExpertOffload::sparse_moe(std::size_t layer, const Tensor& x, const ops::SparseMoeWeights& resident,
                               Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    if (!allocated_pools_) { throw std::logic_error("35B expert offload was not allocated"); }
    if (layer >= kOffloadTextLayers) { throw std::out_of_range("35B expert offload: layer out of range"); }
    if (x.ne[1] > max_tokens_) {
        throw std::invalid_argument("35B expert offload: token count exceeds the planned maximum");
    }
    const Pool& pool = pools_[static_cast<std::size_t>(pool_of_layer_[layer])];
    ops::SparseMoeWeights weights = resident;
    weights.routed_gate_up        = pool.gate_up;
    weights.routed_down           = pool.down;
    ops::SparseMoeExpertResidency residency{
        .host_ids            = static_cast<std::int32_t*>(host_ids_.data()),
        .host_bank_of_expert = static_cast<std::int32_t*>(host_bank_of_expert_.data()),
        .banks               = static_cast<std::int32_t>(pool.slots.size()) + 1,
        .acquire             = [this, layer](std::span<const std::int32_t> selected, std::int32_t* bank_of_expert,
                                 bool cold_allowed, cudaStream_t s) {
            acquire(layer, selected, bank_of_expert, cold_allowed, s);
        },
    };
    if (policy_.host_cold_experts) {
        residency.zero_bank     = static_cast<std::int32_t>(pool.slots.size());
        residency.host_alpha    = static_cast<float*>(host_alpha_.data());
        residency.host_x        = static_cast<std::uint16_t*>(host_x_.data());
        residency.host_cold_sum = static_cast<float*>(host_cold_sum_.data());
        residency.cold_compute  = [this, layer](std::span<const std::int32_t> selected, const float* alpha,
                                               const std::uint16_t* x_bf16, std::int32_t tokens,
                                               const std::int32_t* bank_of_expert, float* cold_sum) {
            cold_compute(layer, selected, alpha, x_bf16, tokens, bank_of_expert, cold_sum);
        };
    }
    ++stats_.calls;
    // NINFER_OFFLOAD_TIMING: synchronize around prefill-size calls to attribute their wall time.
    static const bool timing = std::getenv("NINFER_OFFLOAD_TIMING") != nullptr;
    const bool timed         = timing && x.ne[1] >= 256;
    std::chrono::steady_clock::time_point timed_start;
    if (timed) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        timed_start = std::chrono::steady_clock::now();
    }
    struct TimedScope {
        bool active;
        cudaStream_t stream;
        std::chrono::steady_clock::time_point start;
        double& total;
        ~TimedScope() {
            if (!active) { return; }
            (void)cudaStreamSynchronize(stream);
            total += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
    } timed_scope{timed, stream, timed_start, stats_.prefill_moe_seconds};
    auto scope               = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(ops::sparse_moe_workspace_capacity_bytes(
        weights.routed_gate_up.qtype, weights.routed_down.qtype, x.ne[1], x.ne[1]));
    WorkspaceArena leaf_workspace(storage);
    ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::AddResidual, residual, residency, leaf_workspace, stream);
}

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail
