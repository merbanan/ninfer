#include "targets/qwen3_8_flash_next/impl/expert_cache.h"

#include "artifact/reader.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/expert_cpu.h"
#include "targets/qwen3_8_flash_next/impl/moe_kernels.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

constexpr std::int32_t kHidden       = 2560;
constexpr std::int32_t kIntermediate = 640;
constexpr std::int32_t kTopK         = 10;
constexpr std::int32_t kExperts      = 512;
constexpr std::int32_t kDecodeTokens = 8;
constexpr std::size_t kCopyChunk     = 256U << 10;
constexpr std::uint32_t kMinimumSlots = 64;

std::uint32_t env_u32(const char* name, std::uint32_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') { return fallback; }
    return static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

artifact::BlockScaleBankGeometry bank_geometry(std::uint64_t experts, std::int32_t rows, std::int32_t columns) {
    return artifact::block_scale_bank_geometry(
        artifact::NumericFormat::NVFP4,
        std::array<std::uint64_t, 3>{experts, static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(columns)});
}

HostNvfp4Matrix host_matrix(const Nvfp4ExpertBankView& bank, std::int32_t expert) {
    return {
        .codes   = reinterpret_cast<const std::uint8_t*>(bank.codes) +
                 static_cast<std::size_t>(expert) * bank.code_bytes_per_expert,
        .scales  = reinterpret_cast<const std::uint8_t*>(bank.scales) +
                  static_cast<std::size_t>(expert) * bank.scale_bytes_per_expert,
        .divisor = bank.weight_scale_divisors[expert],
        .rows    = bank.rows,
        .columns = bank.columns,
    };
}

} // namespace

FlashNextOffloadPolicy FlashNextOffloadPolicy::from_environment() {
    FlashNextOffloadPolicy out;
    out.cache_bytes           = static_cast<std::size_t>(env_u32("NINFER_FLASH_NEXT_CACHE_MB", 0)) << 20;
    out.reserve_mib           = env_u32("NINFER_FLASH_NEXT_CACHE_RESERVE_MB", static_cast<std::uint32_t>(out.reserve_mib));
    out.host_max_tokens       = static_cast<std::int32_t>(std::min<std::uint32_t>(
        env_u32("NINFER_FLASH_NEXT_HOST_TOKENS", static_cast<std::uint32_t>(out.host_max_tokens)), kDecodeTokens));
    out.promotions_per_call   = env_u32("NINFER_FLASH_NEXT_PROMOTE", out.promotions_per_call);
    out.admit_misses          = std::max(1U, env_u32("NINFER_FLASH_NEXT_ADMIT", out.admit_misses));
    out.cache_prefill_experts = env_u32("NINFER_FLASH_NEXT_CACHE_PREFILL", out.cache_prefill_experts ? 1 : 0) != 0;
    out.host_threads          = env_u32("NINFER_FLASH_NEXT_CPU_THREADS", out.host_threads);
    if (out.host_threads == 0) { out.host_threads = std::max(1U, std::thread::hardware_concurrency() / 2); }
    out.hybrid_max_tokens = static_cast<std::int32_t>(
        env_u32("NINFER_FLASH_NEXT_HYBRID", static_cast<std::uint32_t>(out.hybrid_max_tokens)));
    if (const char* value = std::getenv("NINFER_FLASH_NEXT_HOST_US"); value != nullptr && value[0] != '\0') {
        char* end              = nullptr;
        out.host_us_per_expert = std::strtod(value, &end);
        if (end != nullptr && *end == ',') { out.host_us_per_column = std::strtod(end + 1, nullptr); }
    }
    out.report = env_u32("NINFER_FLASH_NEXT_OFFLOAD_STATS", 0) != 0;
    return out;
}

FlashNextExpertCache::FlashNextExpertCache(FlashNextOffloadPolicy policy) : policy_(policy) {}

void FlashNextExpertCache::report() const {
    if (stats_.calls == 0) { return; }
    std::fprintf(stderr,
                 "[flash-next offload] calls %llu lookups %llu hit %.1f%% cold %llu uploads %llu "
                 "promotions %llu inserted %llu hybrid calls %llu host %.2fs upload %.2fs\n",
                 static_cast<unsigned long long>(stats_.calls), static_cast<unsigned long long>(stats_.lookups),
                 stats_.lookups ? 100.0 * static_cast<double>(stats_.hits) / static_cast<double>(stats_.lookups) : 0.0,
                 static_cast<unsigned long long>(stats_.cold), static_cast<unsigned long long>(stats_.uploads),
                 static_cast<unsigned long long>(stats_.promotions), static_cast<unsigned long long>(stats_.inserted),
                 static_cast<unsigned long long>(stats_.hybrid_calls), stats_.host_seconds, stats_.upload_seconds);
}

FlashNextExpertCache::~FlashNextExpertCache() {
    if (!allocated_) { return; }
    // Best effort at teardown: the context may already be going away.
    if (copy_stream_ != nullptr) { (void)cudaStreamSynchronize(copy_stream_); }
    for (auto event : promotion_done_) { (void)cudaEventDestroy(event); }
    for (auto event : batch_free_) { (void)cudaEventDestroy(event); }
    if (compute_mark_ != nullptr) { (void)cudaEventDestroy(compute_mark_); }
    if (copy_stream_ != nullptr) { (void)cudaStreamDestroy(copy_stream_); }
}

std::size_t FlashNextExpertCache::packed_bytes() const noexcept {
    const auto align = [](std::size_t v) { return (v + 255U) & ~std::size_t{255U}; };
    return align(gate_.code_bytes) + align(gate_.scale_bytes) + 256 + align(down_.code_bytes) +
           align(down_.scale_bytes) + 256;
}

void FlashNextExpertCache::allocate(const MoeWeights& weights) {
    const auto per_expert = [](const Nvfp4ExpertBankView& bank) {
        return static_cast<std::size_t>(bank.code_bytes_per_expert + bank.scale_bytes_per_expert + sizeof(float));
    };
    const std::size_t expert_bytes = per_expert(weights.expert_gate_up) + per_expert(weights.expert_down);
    std::size_t budget             = policy_.cache_bytes;
    if (budget == 0) {
        std::size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        const std::size_t reserve = (policy_.reserve_mib << 20) + (16U << 20);
        budget = free_bytes > reserve ? free_bytes - reserve : 0;
    }
    const std::size_t fit = budget / expert_bytes;
    if (fit < kMinimumSlots + 1) {
        throw std::runtime_error("Flash-Next expert cache: " + std::to_string(budget >> 20) +
                                 " MiB of device memory hold fewer than " + std::to_string(kMinimumSlots) +
                                 " experts; lower --max-context or the prefill chunk");
    }
    slots_ = static_cast<std::uint32_t>(std::min<std::size_t>(fit - 1, 64U * kExperts));

    const auto describe = [](const artifact::BlockScaleBankGeometry& g, std::int32_t rows, std::int32_t columns) {
        Bank out;
        out.code_bytes     = g.code_plane_bytes / g.experts;
        out.scale_bytes    = g.scale_plane_bytes / g.experts;
        out.scale_offset   = g.scale_plane_offset;
        out.divisor_offset = g.weight_divisor_offset;
        out.total          = g.encoded_bytes;
        out.rows           = rows;
        out.columns        = columns;
        return out;
    };
    gate_      = describe(bank_geometry(slots_ + 1U, 2 * kIntermediate, kHidden), 2 * kIntermediate, kHidden);
    down_      = describe(bank_geometry(slots_ + 1U, kHidden, kIntermediate), kHidden, kIntermediate);
    gate_pool_ = DeviceBuffer(gate_.total);
    down_pool_ = DeviceBuffer(down_.total);
    gate_pool_.fill(0);
    down_pool_.fill(0);
    // The zero expert: codes and scales 0, divisor 1.
    const float one = 1.0F;
    gate_pool_.copy_from_host(&one, sizeof(one), gate_.divisor_offset + sizeof(float) * slots_);
    down_pool_.copy_from_host(&one, sizeof(one), down_.divisor_offset + sizeof(float) * slots_);

    slot_.assign(slots_, Slot{});
    needed_.assign(slots_, 0);
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&compute_mark_, cudaEventDisableTiming));
    for (std::size_t i = 0; i < kPromotionBuffers; ++i) {
        promotion_buffer_[i] = std::make_unique<PinnedHostBuffer>(packed_bytes());
        CUDA_CHECK(cudaEventCreateWithFlags(&promotion_done_[i], cudaEventDisableTiming));
    }
    for (std::size_t i = 0; i < batch_.size(); ++i) {
        batch_[i] = std::make_unique<PinnedHostBuffer>(packed_bytes() * kBatchExperts);
        CUDA_CHECK(cudaEventCreateWithFlags(&batch_free_[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(batch_free_[i], copy_stream_));
    }
    cpu_pool_  = std::make_unique<ops::cpu::SpinPool>(policy_.host_threads);
    allocated_ = true;
    std::fprintf(stderr,
                 "[flash-next offload] device expert cache: %u experts (%.2f GiB), host experts for calls "
                 "<= %d tokens, %u host threads\n",
                 slots_, static_cast<double>(gate_.total + down_.total) / (1U << 30), policy_.host_max_tokens,
                 policy_.host_threads);
}

int FlashNextExpertCache::layer_of(const MoeWeights& weights) {
    const auto [it, inserted] =
        layers_.try_emplace(weights.expert_gate_up.mapped_payload, static_cast<int>(layers_.size()));
    if (inserted) {
        miss_counts_.resize(layers_.size() * kExperts, 0);
        layer_calls_.resize(layers_.size(), 0);
    }
    return it->second;
}

void FlashNextExpertCache::pack(const MoeWeights& weights, std::span<const std::int32_t> experts, std::byte* dst) {
    const auto align = [](std::size_t v) { return (v + 255U) & ~std::size_t{255U}; };
    struct Piece {
        const std::byte* src;
        std::size_t bytes;
        std::size_t offset;
    };
    const auto& g = weights.expert_gate_up;
    const auto& d = weights.expert_down;
    std::vector<Piece> pieces;
    std::size_t at = 0;
    const auto next = [&](const std::byte* src, std::size_t bytes) {
        pieces.push_back({src, bytes, at});
        at += align(bytes);
    };
    for (const std::int32_t expert : experts) {
        at = pieces.size() / 6 * packed_bytes();
        next(g.codes + static_cast<std::size_t>(expert) * g.code_bytes_per_expert, g.code_bytes_per_expert);
        next(g.scales + static_cast<std::size_t>(expert) * g.scale_bytes_per_expert, g.scale_bytes_per_expert);
        next(reinterpret_cast<const std::byte*>(g.weight_scale_divisors + expert), sizeof(float));
        next(d.codes + static_cast<std::size_t>(expert) * d.code_bytes_per_expert, d.code_bytes_per_expert);
        next(d.scales + static_cast<std::size_t>(expert) * d.scale_bytes_per_expert, d.scale_bytes_per_expert);
        next(reinterpret_cast<const std::byte*>(d.weight_scale_divisors + expert), sizeof(float));
    }
    // Chunked so the page faults of a cold mapping are taken by all threads at once.
    std::vector<std::pair<std::size_t, std::size_t>> chunks; // (piece, offset)
    for (std::size_t p = 0; p < pieces.size(); ++p) {
        for (std::size_t o = 0; o < pieces[p].bytes; o += kCopyChunk) { chunks.emplace_back(p, o); }
    }
    cpu_pool_->run(chunks.size(), [&](std::size_t i) {
        const auto& piece = pieces[chunks[i].first];
        const std::size_t o = chunks[i].second;
        std::memcpy(dst + piece.offset + o, piece.src + o, std::min(kCopyChunk, piece.bytes - o));
    });
}

void FlashNextExpertCache::unpack(const std::byte* src, std::byte* gate_base, std::byte* down_base,
                                  std::int64_t index, cudaMemcpyKind kind, cudaStream_t stream) const {
    const auto align = [](std::size_t v) { return (v + 255U) & ~std::size_t{255U}; };
    std::size_t at = 0;
    const auto put = [&](std::byte* dst, std::size_t bytes) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src + at, bytes, kind, stream));
        at += align(bytes);
    };
    put(gate_base + index * gate_.code_bytes, gate_.code_bytes);
    put(gate_base + gate_.scale_offset + index * gate_.scale_bytes, gate_.scale_bytes);
    put(gate_base + gate_.divisor_offset + index * sizeof(float), sizeof(float));
    put(down_base + index * down_.code_bytes, down_.code_bytes);
    put(down_base + down_.scale_offset + index * down_.scale_bytes, down_.scale_bytes);
    put(down_base + down_.divisor_offset + index * sizeof(float), sizeof(float));
}

void FlashNextExpertCache::copy_expert(const std::byte* src_gate, const std::byte* src_down, const Bank& src_gb,
                                       const Bank& src_db, std::int64_t from, std::byte* dst_gate,
                                       std::byte* dst_down, const Bank& dst_gb, const Bank& dst_db,
                                       std::int64_t to, cudaStream_t stream) const {
    const auto copy_bank = [&](const std::byte* src, const Bank& sb, std::byte* dst, const Bank& db) {
        CUDA_CHECK(cudaMemcpyAsync(dst + to * db.code_bytes, src + from * sb.code_bytes, sb.code_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dst + db.scale_offset + to * db.scale_bytes,
                                   src + sb.scale_offset + from * sb.scale_bytes, sb.scale_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dst + db.divisor_offset + to * sizeof(float),
                                   src + sb.divisor_offset + from * sizeof(float), sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
    };
    copy_bank(src_gate, src_gb, dst_gate, dst_gb);
    copy_bank(src_down, src_db, dst_down, dst_db);
}

void FlashNextExpertCache::retire_promotions() {
    for (std::size_t i = 0; i < promotions_.size();) {
        const auto status = cudaEventQuery(promotion_done_[promotions_[i].buffer]);
        if (status == cudaErrorNotReady) {
            ++i;
            continue;
        }
        CUDA_CHECK(status);
        auto& slot   = slot_[promotions_[i].slot];
        slot.pending = false;
        resident_[promotions_[i].key]          = promotions_[i].slot;
        promotion_busy_[promotions_[i].buffer] = false;
        promotions_[i]                         = promotions_.back();
        promotions_.pop_back();
    }
}

void FlashNextExpertCache::evict(std::uint32_t slot) {
    const std::int64_t key = slot_[slot].key;
    if (key < 0) { return; }
    if (const auto it = resident_.find(key); it != resident_.end() && it->second == slot) { resident_.erase(it); }
    slot_[slot].key = -1;
}

std::int64_t FlashNextExpertCache::victim() const {
    std::int64_t best        = -1;
    std::uint64_t best_clock = ~std::uint64_t{0};
    for (std::uint32_t s = 0; s < slots_; ++s) {
        const Slot& slot = slot_[s];
        if (slot.pending || needed_[s] != 0) { continue; }
        if (slot.key < 0) { return s; }
        if (slot.last_used < best_clock) {
            best_clock = slot.last_used;
            best       = s;
        }
    }
    return best;
}

void FlashNextExpertCache::run(const Tensor& input, const MoeWeights& weights, FlashNextMoeWorkspace& scratch,
                               Tensor& output, cudaStream_t stream, void* staging, std::size_t staging_bytes) {
    if (weights.expert_gate_up.experts != kExperts || weights.expert_down.experts != kExperts ||
        weights.expert_gate_up.mapped_payload == nullptr) {
        throw std::runtime_error("Flash-Next expert cache requires the 512-expert host banks");
    }
    if (last_stream_ != nullptr && last_stream_ != stream) { CUDA_CHECK(cudaStreamSynchronize(last_stream_)); }
    last_stream_ = stream;
    if (!allocated_) { allocate(weights); }
    const int layer = layer_of(weights);
    ++stats_.calls;
    ++clock_;
    if (input.ne[1] <= policy_.host_max_tokens) {
        decode(layer, input, weights, scratch, output, stream);
    } else {
        prefill(layer, input, weights, scratch, output, stream, staging, staging_bytes);
    }
}

void FlashNextExpertCache::decode(int layer, const Tensor& input, const MoeWeights& weights,
                                  FlashNextMoeWorkspace& scratch, Tensor& output, cudaStream_t stream) {
    const std::int32_t tokens = input.ne[1];
    auto* ids     = static_cast<std::int32_t*>(host_ids_.data());
    auto* alpha   = static_cast<float*>(host_alpha_.data());
    auto* x       = static_cast<std::uint16_t*>(host_x_.data());
    const auto n  = static_cast<std::size_t>(tokens) * kTopK;
    CUDA_CHECK(cudaMemcpyAsync(ids, scratch.ids.data, n * sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(alpha, scratch.alpha.data, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(x, input.data, static_cast<std::size_t>(tokens) * kHidden * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    retire_promotions();

    // Distinct experts of the call, with their routed (column, weight) pairs.
    std::array<std::int32_t, kExperts> index_of;
    index_of.fill(-1);
    std::vector<FlashNextColdExpert> cold;
    std::vector<std::int32_t> slot_of_expert;
    std::vector<std::int32_t> experts;
    for (std::size_t i = 0; i < n; ++i) {
        const std::int32_t e = ids[i];
        if (e < 0 || e >= kExperts) { throw std::runtime_error("Flash-Next router produced an out-of-range expert id"); }
        if (index_of[e] >= 0) { continue; }
        index_of[e] = static_cast<std::int32_t>(experts.size());
        experts.push_back(e);
        const std::int64_t key = static_cast<std::int64_t>(layer) * kExperts + e;
        const auto it          = resident_.find(key);
        if (it != resident_.end()) {
            slot_of_expert.push_back(static_cast<std::int32_t>(it->second));
            slot_[it->second].last_used = clock_;
            needed_[it->second]         = 1;
        } else {
            slot_of_expert.push_back(-1 - static_cast<std::int32_t>(cold.size()));
            FlashNextColdExpert work;
            work.gate_up = host_matrix(weights.expert_gate_up, e);
            work.down    = host_matrix(weights.expert_down, e);
            cold.push_back(std::move(work));
        }
    }
    stats_.lookups += experts.size();
    stats_.hits += experts.size() - cold.size();
    stats_.cold += cold.size();
    for (std::size_t i = 0; i < n; ++i) {
        const std::int32_t slot = slot_of_expert[static_cast<std::size_t>(index_of[ids[i]])];
        if (slot < 0) {
            auto& work = cold[static_cast<std::size_t>(-1 - slot)];
            work.columns.push_back(static_cast<std::int32_t>(i / kTopK));
            work.weights.push_back(alpha[i]);
        }
        ids[i] = slot >= 0 ? slot : static_cast<std::int32_t>(slots_);
    }
    CUDA_CHECK(cudaMemcpyAsync(scratch.ids.data, ids, n * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
    scratch.cold_sum = nullptr;
    if (!cold.empty()) {
        const auto start = std::chrono::steady_clock::now();
        auto* sum        = static_cast<float*>(host_cold_.data());
        flash_next_cold_experts(cold, x, tokens, sum, *cpu_pool_);
        stats_.host_seconds += seconds_since(start);
        CUDA_CHECK(cudaMemcpyAsync(device_cold_.p, sum, static_cast<std::size_t>(tokens) * kHidden * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        scratch.cold_sum = static_cast<const float*>(device_cold_.p);
    }
    MoeWeights cached     = weights;
    cached.expert_gate_up = make_nvfp4_expert_bank_view(gate_pool_.p, gate_.total, static_cast<std::int32_t>(slots_ + 1),
                                                        2 * kIntermediate, kHidden);
    cached.expert_down    = make_nvfp4_expert_bank_view(down_pool_.p, down_.total, static_cast<std::int32_t>(slots_ + 1),
                                                        kHidden, kIntermediate);
    flash_next_moe_kernels_launch(input, cached, scratch, output, stream);
    scratch.cold_sum = nullptr;

    // Background promotions of the most frequently missed experts.
    if (!cold.empty() && policy_.promotions_per_call > 0) {
        std::vector<std::pair<std::uint16_t, std::int32_t>> candidates; // (misses, expert)
        for (std::size_t i = 0; i < experts.size(); ++i) {
            if (slot_of_expert[i] >= 0) { continue; }
            auto& misses = miss_counts_[static_cast<std::size_t>(layer) * kExperts + experts[i]];
            misses = static_cast<std::uint16_t>(std::min<int>(misses + 1, 0xFFFF));
            if (misses >= policy_.admit_misses) { candidates.emplace_back(misses, experts[i]); }
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::uint32_t issued = 0;
        bool marked          = false;
        for (const auto& [misses, e] : candidates) {
            if (issued >= policy_.promotions_per_call) { break; }
            const std::int64_t key = static_cast<std::int64_t>(layer) * kExperts + e;
            bool in_flight         = false;
            for (const auto& p : promotions_) { in_flight = in_flight || p.key == key; }
            if (in_flight) { continue; }
            std::size_t buffer = kPromotionBuffers;
            for (std::size_t b = 0; b < kPromotionBuffers; ++b) {
                if (!promotion_busy_[b]) {
                    buffer = b;
                    break;
                }
            }
            const std::int64_t v = victim();
            if (buffer == kPromotionBuffers || v < 0) { break; }
            auto& slot = slot_[static_cast<std::size_t>(v)];
            evict(static_cast<std::uint32_t>(v));
            slot.key       = key;
            slot.pending   = true;
            slot.last_used = clock_;
            auto* packed   = static_cast<std::byte*>(promotion_buffer_[buffer]->data());
            pack(weights, std::span<const std::int32_t>(&e, 1), packed);
            if (!marked) {
                CUDA_CHECK(cudaEventRecord(compute_mark_, stream));
                CUDA_CHECK(cudaStreamWaitEvent(copy_stream_, compute_mark_, 0));
                marked = true;
            }
            unpack(packed, static_cast<std::byte*>(gate_pool_.p), static_cast<std::byte*>(down_pool_.p), v,
                   cudaMemcpyHostToDevice, copy_stream_);
            CUDA_CHECK(cudaEventRecord(promotion_done_[buffer], copy_stream_));
            promotion_busy_[buffer] = true;
            promotions_.push_back({static_cast<std::uint32_t>(v), key, buffer});
            miss_counts_[static_cast<std::size_t>(layer) * kExperts + e] = 0;
            ++stats_.promotions;
            ++issued;
        }
    }
    if (++layer_calls_[static_cast<std::size_t>(layer)] % 64 == 0) {
        for (int e = 0; e < kExperts; ++e) { miss_counts_[static_cast<std::size_t>(layer) * kExperts + e] /= 2; }
    }
    for (const auto slot : slot_of_expert) {
        if (slot >= 0) { needed_[static_cast<std::size_t>(slot)] = 0; }
    }
}

void FlashNextExpertCache::prefill(int layer, const Tensor& input, const MoeWeights& weights,
                                   FlashNextMoeWorkspace& scratch, Tensor& output, cudaStream_t stream,
                                   void* staging, std::size_t staging_bytes) {
    const std::int32_t tokens = input.ne[1];
    const auto n              = static_cast<std::size_t>(tokens) * kTopK;
    const auto grow = [](std::unique_ptr<PinnedHostBuffer>& buffer, std::size_t bytes) {
        if (!buffer || buffer->size() < bytes) { buffer = std::make_unique<PinnedHostBuffer>(bytes); }
        return buffer->data();
    };
    auto* ids = static_cast<std::int32_t*>(grow(prefill_ids_, n * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMemcpyAsync(ids, scratch.ids.data, n * sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    retire_promotions();

    std::array<std::int32_t, kExperts> columns_of{};
    for (std::size_t i = 0; i < n; ++i) {
        if (ids[i] < 0 || ids[i] >= kExperts) {
            throw std::runtime_error("Flash-Next router produced an out-of-range expert id");
        }
        ++columns_of[ids[i]];
    }
    // Device experts: cached ones (copied from their slot) and uploads; the host takes the
    // missing experts with the fewest columns while that shortens the call (see the policy).
    std::vector<std::int32_t> device;             // experts in staging order
    std::vector<std::int64_t> source;             // pool slot of device[i], or -1 = upload
    std::vector<std::int32_t> missing;
    for (std::int32_t e = 0; e < kExperts; ++e) {
        if (columns_of[e] == 0) { continue; }
        const auto it = resident_.find(static_cast<std::int64_t>(layer) * kExperts + e);
        if (it != resident_.end()) {
            device.push_back(e);
            source.push_back(it->second);
        } else {
            missing.push_back(e);
        }
    }
    std::size_t host_count = 0;
    if (!missing.empty() && tokens <= policy_.hybrid_max_tokens) {
        std::stable_sort(missing.begin(), missing.end(),
                         [&](std::int32_t x, std::int32_t y) { return columns_of[x] < columns_of[y]; });
        const double upload_us = 1e-3 * static_cast<double>(packed_bytes()) / policy_.upload_gbps;
        double best            = static_cast<double>(missing.size()) *
                      std::max(upload_us, policy_.pack_us_per_expert);
        double host_us         = 0.0;
        for (std::size_t k = 1; k <= missing.size(); ++k) {
            host_us += policy_.host_us_per_expert +
                       policy_.host_us_per_column * static_cast<double>(columns_of[missing[k - 1]] - 1);
            const auto rest   = static_cast<double>(missing.size() - k);
            const double cost = std::max(host_us + rest * policy_.pack_us_per_expert, rest * upload_us);
            if (cost < best) {
                best       = cost;
                host_count = k;
            }
        }
    }
    const std::size_t first_upload = device.size();
    for (std::size_t i = host_count; i < missing.size(); ++i) {
        device.push_back(missing[i]);
        source.push_back(-1);
    }
    const std::size_t zero_index = device.size();
    const auto count = static_cast<std::uint64_t>(device.size() + (host_count > 0 ? 1U : 0U));
    std::array<std::int32_t, kExperts> local_of;
    local_of.fill(static_cast<std::int32_t>(zero_index));
    for (std::size_t i = 0; i < device.size(); ++i) { local_of[device[i]] = static_cast<std::int32_t>(i); }

    const auto gate_g  = bank_geometry(count, 2 * kIntermediate, kHidden);
    const auto down_g  = bank_geometry(count, kHidden, kIntermediate);
    const std::size_t down_offset = (gate_g.encoded_bytes + 255U) & ~std::size_t{255U};
    if (count > kExperts || staging == nullptr || down_offset + down_g.encoded_bytes > staging_bytes) {
        throw std::runtime_error("Flash-Next expert staging reservation is too small");
    }
    const auto describe = [](const artifact::BlockScaleBankGeometry& g) {
        Bank out;
        out.code_bytes     = g.code_plane_bytes / g.experts;
        out.scale_bytes    = g.scale_plane_bytes / g.experts;
        out.scale_offset   = g.scale_plane_offset;
        out.divisor_offset = g.weight_divisor_offset;
        out.total          = g.encoded_bytes;
        return out;
    };
    const Bank stage_gate = describe(gate_g);
    const Bank stage_down = describe(down_g);
    auto* stage_gate_base = static_cast<std::byte*>(staging);
    auto* stage_down_base = stage_gate_base + down_offset;
    auto* pool_gate       = static_cast<std::byte*>(gate_pool_.p);
    auto* pool_down       = static_cast<std::byte*>(down_pool_.p);

    // Host experts need the input and route weights; fetched while the stream is idle.
    std::uint16_t* x = nullptr;
    float* alpha     = nullptr;
    if (host_count > 0) {
        x     = static_cast<std::uint16_t*>(grow(prefill_x_, static_cast<std::size_t>(tokens) * kHidden * 2));
        alpha = static_cast<float*>(grow(prefill_alpha_, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(x, input.data, static_cast<std::size_t>(tokens) * kHidden * 2,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(alpha, scratch.alpha.data, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        copy_expert(pool_gate, pool_down, gate_, down_, slots_, stage_gate_base, stage_down_base, stage_gate,
                    stage_down, static_cast<std::int64_t>(zero_index), stream);
    }

    std::vector<std::uint32_t> used;
    for (std::size_t i = 0; i < first_upload; ++i) {
        const auto slot = static_cast<std::uint32_t>(source[i]);
        slot_[slot].last_used = clock_;
        needed_[slot]         = 1;
        used.push_back(slot);
        copy_expert(pool_gate, pool_down, gate_, down_, slot, stage_gate_base, stage_down_base, stage_gate,
                    stage_down, static_cast<std::int64_t>(i), stream);
    }
    stats_.lookups += first_upload + missing.size();
    stats_.hits += first_upload;
    stats_.cold += host_count;
    stats_.uploads += missing.size() - host_count;
    if (host_count > 0) { ++stats_.hybrid_calls; }

    // Uploads in batches through two alternating pinned buffers.
    const auto start      = std::chrono::steady_clock::now();
    const std::size_t per = packed_bytes();
    const auto align      = [](std::size_t v) { return (v + 255U) & ~std::size_t{255U}; };
    for (std::size_t first = first_upload, batch = 0; first < device.size(); first += kBatchExperts, ++batch) {
        const std::size_t b = batch % batch_.size();
        CUDA_CHECK(cudaEventSynchronize(batch_free_[b]));
        auto* buffer = static_cast<std::byte*>(batch_[b]->data());
        const std::size_t in_batch = std::min(kBatchExperts, device.size() - first);
        pack(weights, std::span<const std::int32_t>(device.data() + first, in_batch), buffer);
        for (std::size_t j = 0; j < in_batch; ++j) {
            const std::byte* src  = buffer + j * per;
            const auto to         = static_cast<std::int64_t>(first + j);
            std::size_t at        = 0;
            const auto put = [&](std::byte* dst, std::size_t bytes) {
                CUDA_CHECK(cudaMemcpyAsync(dst, src + at, bytes, cudaMemcpyHostToDevice, stream));
                at += align(bytes);
            };
            put(stage_gate_base + to * stage_gate.code_bytes, stage_gate.code_bytes);
            put(stage_gate_base + stage_gate.scale_offset + to * stage_gate.scale_bytes, stage_gate.scale_bytes);
            put(stage_gate_base + stage_gate.divisor_offset + to * sizeof(float), sizeof(float));
            put(stage_down_base + to * stage_down.code_bytes, stage_down.code_bytes);
            put(stage_down_base + stage_down.scale_offset + to * stage_down.scale_bytes, stage_down.scale_bytes);
            put(stage_down_base + stage_down.divisor_offset + to * sizeof(float), sizeof(float));
        }
        CUDA_CHECK(cudaEventRecord(batch_free_[b], stream));
    }
    stats_.upload_seconds += seconds_since(start);

    // Host work lists come from the original ids, before the remap below.
    std::vector<FlashNextColdExpert> cold(host_count);
    if (host_count > 0) {
        std::array<std::int32_t, kExperts> work_of;
        work_of.fill(-1);
        for (std::size_t k = 0; k < host_count; ++k) {
            work_of[missing[k]]  = static_cast<std::int32_t>(k);
            cold[k].gate_up      = host_matrix(weights.expert_gate_up, missing[k]);
            cold[k].down         = host_matrix(weights.expert_down, missing[k]);
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (const std::int32_t w = work_of[ids[i]]; w >= 0) {
                cold[static_cast<std::size_t>(w)].columns.push_back(static_cast<std::int32_t>(i / kTopK));
                cold[static_cast<std::size_t>(w)].weights.push_back(alpha[i]);
            }
        }
    }
    for (std::size_t i = 0; i < n; ++i) { ids[i] = local_of[ids[i]]; }
    CUDA_CHECK(cudaMemcpyAsync(scratch.ids.data, ids, n * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
    MoeWeights staged     = weights;
    staged.expert_gate_up = make_nvfp4_expert_bank_view(stage_gate_base, gate_g.encoded_bytes,
                                                        static_cast<std::int32_t>(count), 2 * kIntermediate, kHidden);
    staged.expert_down    = make_nvfp4_expert_bank_view(stage_down_base, down_g.encoded_bytes,
                                                        static_cast<std::int32_t>(count), kHidden, kIntermediate);
    scratch.cold_sum = nullptr;
    flash_next_moe_kernels_launch(input, staged, scratch, output, stream);

    // The host experts run while the device computes the rest.
    if (host_count > 0) {
        const auto host_start = std::chrono::steady_clock::now();
        const std::size_t sum_bytes = static_cast<std::size_t>(tokens) * kHidden * sizeof(float);
        auto* sum = static_cast<float*>(grow(prefill_cold_, sum_bytes));
        flash_next_cold_experts(cold, x, tokens, sum, *cpu_pool_);
        stats_.host_seconds += seconds_since(host_start);
        if (prefill_cold_device_.bytes < sum_bytes) { prefill_cold_device_ = DeviceBuffer(sum_bytes); }
        CUDA_CHECK(cudaMemcpyAsync(prefill_cold_device_.p, sum, sum_bytes, cudaMemcpyHostToDevice, stream));
        flash_next_moe_add_cold(output, static_cast<const float*>(prefill_cold_device_.p), tokens, stream);
    }

    // Keep the uploaded experts: device copies from staging into the least recently used slots.
    if (policy_.cache_prefill_experts) {
        for (std::size_t local = first_upload; local < device.size(); ++local) {
            const std::int64_t v = victim();
            if (v < 0) { break; }
            const std::int64_t key = static_cast<std::int64_t>(layer) * kExperts + device[local];
            bool in_flight         = false;
            for (const auto& p : promotions_) { in_flight = in_flight || p.key == key; }
            if (in_flight) { continue; }
            evict(static_cast<std::uint32_t>(v));
            auto& slot     = slot_[static_cast<std::size_t>(v)];
            slot.key       = key;
            slot.last_used = clock_;
            needed_[static_cast<std::size_t>(v)] = 1;
            used.push_back(static_cast<std::uint32_t>(v));
            copy_expert(stage_gate_base, stage_down_base, stage_gate, stage_down, static_cast<std::int64_t>(local),
                        pool_gate, pool_down, gate_, down_, v, stream);
            resident_[key] = static_cast<std::uint32_t>(v);
            ++stats_.inserted;
        }
    }
    for (const auto s : used) { needed_[s] = 0; }
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
