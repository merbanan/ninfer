#include "ops/sparse_moe/cpu/sparse_moe_cpu.h"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ninfer::ops::cpu {
namespace {

constexpr int kHidden       = 2048;
constexpr int kIntermediate = 512;
constexpr int kGroup        = 64;
constexpr int kMaxColumns   = 16; // columns one weight decode is reused across
constexpr int kGateRowBlock = 64;
constexpr int kDownRowBlock = 64;

float half_to_float(std::uint16_t h) {
    const std::uint32_t sign     = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t exponent = (h >> 10) & 0x1Fu;
    const std::uint32_t mantissa = h & 0x3FFu;
    std::uint32_t bits           = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            float value = std::ldexp(static_cast<float>(mantissa), -24);
            std::memcpy(&bits, &value, 4);
            bits |= sign;
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float out = 0;
    std::memcpy(&out, &bits, 4);
    return out;
}

float bf16_to_float(std::uint16_t h) {
    const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float out                = 0;
    std::memcpy(&out, &bits, 4);
    return out;
}

// Signed code of element `index` of one group (the tests' row-split reference rule).
int scalar_code(const std::uint8_t* codes, const std::uint8_t* high, int bits, int index) {
    const std::uint8_t byte = codes[index >> 1];
    std::uint32_t u         = (index & 1) ? (byte >> 4) : (byte & 0x0Fu);
    if (bits == 5) { u |= ((high[index >> 3] >> (index & 7)) & 1u) << 4; }
    if (bits == 6) { u |= ((high[(index * 2) >> 3] >> ((index * 2) & 7)) & 3u) << 4; }
    const std::uint32_t sign = 1u << (bits - 1);
    return (u & sign) ? static_cast<int>(u) - static_cast<int>(1u << bits) : static_cast<int>(u);
}

// Row-split planes of one matrix and its row geometry.
struct Rows {
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const std::uint8_t* scales;
    int columns; // K
    int bits;    // 4, 5 or 6
    [[nodiscard]] int groups() const { return columns / kGroup; }
    [[nodiscard]] int high_bytes() const { return bits == 4 ? 0 : kGroup * (bits - 4) / 8; }
};

void scalar_rows(const Rows& m, int row_begin, int row_end, const float* const* x, int ncols,
                 float* out, int out_stride, int out_row0) {
    for (int row = row_begin; row < row_end; ++row) {
        float acc[kMaxColumns] = {};
        for (int g = 0; g < m.groups(); ++g) {
            const std::int64_t group = static_cast<std::int64_t>(row) * m.groups() + g;
            const std::uint8_t* codes = m.codes + group * (kGroup / 2);
            const std::uint8_t* high  = m.high != nullptr ? m.high + group * m.high_bytes() : nullptr;
            std::uint16_t scale_bits  = 0;
            std::memcpy(&scale_bits, m.scales + group * 2, 2);
            const float scale = half_to_float(scale_bits);
            for (int i = 0; i < kGroup; ++i) {
                const float w = static_cast<float>(scalar_code(codes, high, m.bits, i)) * scale;
                for (int c = 0; c < ncols; ++c) { acc[c] += w * x[c][g * kGroup + i]; }
            }
        }
        for (int c = 0; c < ncols; ++c) { out[static_cast<std::size_t>(c) * out_stride + row - out_row0] = acc[c]; }
    }
}

__attribute__((target("avx2,fma,f16c"))) inline __m256i expand_bits_q5(const std::uint8_t* h4) {
    // byte j of the result = 0x10 when bit j of the 32-bit little-endian word h4 is set.
    std::int32_t word = 0;
    std::memcpy(&word, h4, 4);
    const __m256i v       = _mm256_set1_epi32(word);
    const __m256i shuffle = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                                             2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bit = _mm256_set1_epi64x(static_cast<long long>(0x8040201008040201ULL));
    const __m256i m   = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(v, shuffle), bit), bit);
    return _mm256_and_si256(m, _mm256_set1_epi8(0x10));
}

__attribute__((target("avx2,fma,f16c"))) inline __m256i expand_bits_q6(const std::uint8_t* h8) {
    // byte j of the result = (2-bit field j of the 64-bit word h8) << 4.
    long long word = 0;
    std::memcpy(&word, h8, 8);
    const __m256i v       = _mm256_set1_epi64x(word);
    const __m256i shuffle = _mm256_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5,
                                             5, 5, 6, 6, 6, 6, 7, 7, 7, 7);
    const __m256i s    = _mm256_shuffle_epi8(v, shuffle);
    const __m256i low  = _mm256_set1_epi32(0x40100401);
    const __m256i high = _mm256_set1_epi32(static_cast<int>(0x80200802u));
    const __m256i b0   = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(s, low), low), _mm256_set1_epi8(0x10));
    const __m256i b1 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(s, high), high), _mm256_set1_epi8(0x20));
    return _mm256_or_si256(b0, b1);
}

// Decodes one 64-element group into w[0..7] (8 floats each, natural order), times its scale.
__attribute__((target("avx2,fma,f16c"))) inline void decode_group(const Rows& m, std::int64_t group, __m256 (&w)[8]) {
    const __m256i b    = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m.codes + group * (kGroup / 2)));
    const __m256i mask = _mm256_set1_epi8(0x0F);
    const __m256i lo   = _mm256_and_si256(b, mask);
    const __m256i hi   = _mm256_and_si256(_mm256_srli_epi16(b, 4), mask);
    const __m256i u0   = _mm256_unpacklo_epi8(lo, hi);
    const __m256i u1   = _mm256_unpackhi_epi8(lo, hi);
    __m256i n[2]       = {_mm256_permute2x128_si256(u0, u1, 0x20), _mm256_permute2x128_si256(u0, u1, 0x31)};
    int offset         = 8;
    if (m.bits == 5) {
        const std::uint8_t* h = m.high + group * 8;
        n[0]                  = _mm256_or_si256(n[0], expand_bits_q5(h));
        n[1]                  = _mm256_or_si256(n[1], expand_bits_q5(h + 4));
        offset                = 16;
    } else if (m.bits == 6) {
        const std::uint8_t* h = m.high + group * 16;
        n[0]                  = _mm256_or_si256(n[0], expand_bits_q6(h));
        n[1]                  = _mm256_or_si256(n[1], expand_bits_q6(h + 8));
        offset                = 32;
    }
    const __m256i bias = _mm256_set1_epi8(static_cast<char>(offset));
    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, m.scales + group * 2, 2);
    const __m256 scale = _mm256_set1_ps(_cvtsh_ss(scale_bits));
    for (int half = 0; half < 2; ++half) {
        const __m256i s    = _mm256_sub_epi8(_mm256_xor_si256(n[half], bias), bias);
        const __m128i lo16 = _mm256_castsi256_si128(s);
        const __m128i hi16 = _mm256_extracti128_si256(s, 1);
        w[half * 4 + 0] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo16)), scale);
        w[half * 4 + 1] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo16, 8))), scale);
        w[half * 4 + 2] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi16)), scale);
        w[half * 4 + 3] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi16, 8))), scale);
    }
}

__attribute__((target("avx2,fma,f16c"))) inline float hsum(__m256 v) {
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 t = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(t, _mm_movehdup_ps(t)));
}

__attribute__((target("avx2,fma,f16c"))) void avx2_rows(const Rows& m, int row_begin, int row_end,
                                                   const float* const* x, int ncols, float* out, int out_stride,
                                                   int out_row0) {
    for (int row = row_begin; row < row_end; ++row) {
        __m256 acc[kMaxColumns];
        for (int c = 0; c < ncols; ++c) { acc[c] = _mm256_setzero_ps(); }
        for (int g = 0; g < m.groups(); ++g) {
            __m256 w[8];
            decode_group(m, static_cast<std::int64_t>(row) * m.groups() + g, w);
            for (int c = 0; c < ncols; ++c) {
                const float* xc = x[c] + g * kGroup;
                for (int i = 0; i < 8; ++i) { acc[c] = _mm256_fmadd_ps(w[i], _mm256_loadu_ps(xc + 8 * i), acc[c]); }
            }
        }
        for (int c = 0; c < ncols; ++c) { out[static_cast<std::size_t>(c) * out_stride + row - out_row0] = hsum(acc[c]); }
    }
}

bool cpu_has_avx2() {
    static const bool has = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
                            __builtin_cpu_supports("f16c");
    return has;
}

// out[c * out_stride + (row - out_row0)] = row `row` of m times x[c], for rows in [row_begin, row_end).
void rows(const Rows& m, int row_begin, int row_end, const float* const* x, int ncols, float* out, int out_stride,
          int out_row0 = 0) {
    if (cpu_has_avx2()) {
        avx2_rows(m, row_begin, row_end, x, ncols, out, out_stride, out_row0);
    } else {
        scalar_rows(m, row_begin, row_end, x, ncols, out, out_stride, out_row0);
    }
}

int bits_of(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
        return 4;
    case QType::Q5G64_F16S:
        return 5;
    case QType::Q6G64_F16S:
        return 6;
    default:
        throw std::invalid_argument("sparse_moe cpu: routed down must be Q5 or Q6");
    }
}

} // namespace

SpinPool::SpinPool(unsigned threads) {
    for (unsigned i = 1; i < std::max(1u, threads); ++i) { workers_.emplace_back([this] { worker_loop(); }); }
}

SpinPool::~SpinPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_.store(true);
        generation_.fetch_add(1, std::memory_order_release);
    }
    wake_.notify_all();
    for (auto& worker : workers_) { worker.join(); }
}

void SpinPool::worker_loop() {
    std::uint64_t seen = 0;
    for (;;) {
        // Spin for a short while (decode issues regions back to back), then sleep.
        const auto spin_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(5);
        std::uint64_t generation = generation_.load(std::memory_order_acquire);
        while (generation == seen && std::chrono::steady_clock::now() < spin_until) {
            _mm_pause();
            generation = generation_.load(std::memory_order_acquire);
        }
        if (generation == seen) {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] { return generation_.load(std::memory_order_acquire) != seen; });
            generation = generation_.load(std::memory_order_acquire);
        }
        seen = generation;
        if (stop_.load()) { return; }
        const std::size_t count = count_;
        const auto* task        = task_;
        for (std::size_t i = next_.fetch_add(1); i < count; i = next_.fetch_add(1)) {
            (*task)(i);
            done_.fetch_add(1, std::memory_order_release);
        }
        active_.fetch_sub(1, std::memory_order_acq_rel);
    }
}

void SpinPool::run(std::size_t count, const std::function<void(std::size_t)>& task) {
    if (count == 0) { return; }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        count_ = count;
        task_  = &task;
        next_.store(0);
        done_.store(0);
        active_.store(workers_.size(), std::memory_order_release);
        generation_.fetch_add(1, std::memory_order_release);
    }
    wake_.notify_all();
    for (std::size_t i = next_.fetch_add(1); i < count; i = next_.fetch_add(1)) {
        task(i);
        done_.fetch_add(1, std::memory_order_release);
    }
    // Every worker leaves this generation before the next run may reset the shared counters.
    while (done_.load(std::memory_order_acquire) < count || active_.load(std::memory_order_acquire) != 0) {
        _mm_pause();
    }
}

void cold_experts(std::span<const ColdExpertWork> work, const std::uint16_t* x_bf16, std::int32_t tokens,
                  float* out, SpinPool& pool) {
    std::fill(out, out + static_cast<std::size_t>(tokens) * kHidden, 0.0F);
    if (work.empty()) { return; }
    // FP32 inputs, then per work item: gate/up outputs [columns][1024] and activations [columns][512].
    std::vector<float> x(static_cast<std::size_t>(tokens) * kHidden);
    for (std::size_t i = 0; i < x.size(); ++i) { x[i] = bf16_to_float(x_bf16[i]); }
    std::vector<std::size_t> base(work.size() + 1, 0);
    for (std::size_t w = 0; w < work.size(); ++w) {
        if (work[w].columns.size() != work[w].weights.size() || work[w].columns.empty()) {
            throw std::invalid_argument("sparse_moe cpu: invalid cold expert work");
        }
        base[w + 1] = base[w] + work[w].columns.size();
    }
    std::vector<float> gate_up(base.back() * 2 * kIntermediate);
    std::vector<float> act(base.back() * kIntermediate);

    constexpr int kGateBlocks = 2 * kIntermediate / kGateRowBlock;
    pool.run(work.size() * kGateBlocks, [&](std::size_t task) {
        const std::size_t w     = task / kGateBlocks;
        const int block         = static_cast<int>(task % kGateBlocks);
        const ColdExpertWork& e = work[w];
        const Rows m{e.expert.gate_codes, nullptr, e.expert.gate_scales, kHidden, 4};
        // Columns go through in chunks so one weight decode serves up to kMaxColumns of them.
        for (std::size_t c0 = 0; c0 < e.columns.size(); c0 += kMaxColumns) {
            const std::size_t n = std::min<std::size_t>(kMaxColumns, e.columns.size() - c0);
            const float* xs[kMaxColumns];
            for (std::size_t c = 0; c < n; ++c) {
                xs[c] = x.data() + static_cast<std::size_t>(e.columns[c0 + c]) * kHidden;
            }
            rows(m, block * kGateRowBlock, (block + 1) * kGateRowBlock, xs, static_cast<int>(n),
                 gate_up.data() + (base[w] + c0) * 2 * kIntermediate, 2 * kIntermediate);
        }
    });
    for (std::size_t i = 0; i < base.back(); ++i) {
        const float* gu = gate_up.data() + i * 2 * kIntermediate;
        float* a        = act.data() + i * kIntermediate;
        for (int j = 0; j < kIntermediate; ++j) {
            const float g = gu[j];
            a[j]          = g / (1.0F + std::exp(-g)) * gu[kIntermediate + j];
        }
    }
    constexpr int kDownBlocks = kHidden / kDownRowBlock;
    pool.run(kDownBlocks, [&](std::size_t block) {
        const int row_begin = static_cast<int>(block) * kDownRowBlock;
        float partial[kMaxColumns * kDownRowBlock];
        for (std::size_t w = 0; w < work.size(); ++w) {
            const ColdExpertWork& e = work[w];
            const Rows m{e.expert.down_codes, e.expert.down_high, e.expert.down_scales, kIntermediate,
                         bits_of(e.expert.down_qtype)};
            for (std::size_t c0 = 0; c0 < e.columns.size(); c0 += kMaxColumns) {
                const std::size_t n = std::min<std::size_t>(kMaxColumns, e.columns.size() - c0);
                const float* as[kMaxColumns];
                for (std::size_t c = 0; c < n; ++c) { as[c] = act.data() + (base[w] + c0 + c) * kIntermediate; }
                rows(m, row_begin, row_begin + kDownRowBlock, as, static_cast<int>(n), partial, kDownRowBlock,
                     row_begin);
                for (std::size_t c = 0; c < n; ++c) {
                    float* o = out + static_cast<std::size_t>(e.columns[c0 + c]) * kHidden + row_begin;
                    for (int r = 0; r < kDownRowBlock; ++r) {
                        o[r] += e.weights[c0 + c] * partial[c * kDownRowBlock + r];
                    }
                }
            }
        }
    });
}

} // namespace ninfer::ops::cpu
