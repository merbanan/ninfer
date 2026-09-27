#include "targets/qwen3_8_flash_next/impl/expert_cpu.h"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

constexpr int kHidden       = 2560;
constexpr int kIntermediate = 640;
constexpr int kGroup        = 16;  // columns per E4M3 scale
constexpr int kMaxColumns   = 16;  // token columns one weight decode serves
constexpr int kRowBlock     = 64;

// E2M1 magnitudes of codes 0..7 (bit 3 is the sign).
constexpr std::array<float, 8> kE2m1 = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};

float e4m3_value(std::uint8_t bits) {
    const int sign     = bits >> 7;
    const int exponent = (bits >> 3) & 0xF;
    const int mantissa = bits & 0x7;
    float value        = 0.0F;
    if (exponent == 0xF && mantissa == 0x7) {
        value = NAN;
    } else if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa) / 8.0F, -6);
    } else {
        value = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, exponent - 7);
    }
    return sign ? -value : value;
}

const std::array<float, 256>& e4m3_table() {
    static const std::array<float, 256> table = [] {
        std::array<float, 256> out{};
        for (int i = 0; i < 256; ++i) { out[static_cast<std::size_t>(i)] = e4m3_value(static_cast<std::uint8_t>(i)); }
        return out;
    }();
    return table;
}

// Byte offset of (row, group) in the swizzled scale plane (storage-layouts.md, section 4).
std::size_t scale_offset(std::int32_t row, std::int32_t group, std::int32_t columns) {
    const std::int32_t k_tiles    = columns / 64;
    const std::int32_t row_tile   = row / 128;
    const std::int32_t row_inner  = row % 128;
    const std::int32_t scale_tile = group / 4;
    const std::int32_t scale_lane = group % 4;
    return static_cast<std::size_t>(row_tile * k_tiles + scale_tile) * 512 +
           static_cast<std::size_t>(row_inner % 32) * 16 + static_cast<std::size_t>(row_inner / 32) * 4 +
           static_cast<std::size_t>(scale_lane);
}

float bf16_to_float(std::uint16_t h) {
    const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float out                = 0.0F;
    std::memcpy(&out, &bits, 4);
    return out;
}

// Round-to-nearest-even FP32 -> BF16 -> FP32 (finite inputs).
float round_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    bits += 0x7FFFU + ((bits >> 16) & 1U);
    bits &= 0xFFFF0000U;
    float out = 0.0F;
    std::memcpy(&out, &bits, 4);
    return out;
}

// out[c * out_stride + (row - out_row0)] = row `row` of m times x[c] for rows [row_begin, row_end).
__attribute__((target("avx2,fma"))) void rows_avx2(const HostNvfp4Matrix& m, int row_begin, int row_end,
                                                   const float* const* x, int ncols, float* out, int out_stride,
                                                   int out_row0) {
    const auto& e4m3      = e4m3_table();
    const float inverse   = 1.0F / m.divisor;
    const __m256 lut      = _mm256_loadu_ps(kE2m1.data());
    const __m256i low7    = _mm256_set1_epi32(7);
    const __m256i sign8   = _mm256_set1_epi32(8);
    const int groups      = m.columns / kGroup;
    for (int row = row_begin; row < row_end; ++row) {
        __m256 acc[kMaxColumns];
        for (int c = 0; c < ncols; ++c) { acc[c] = _mm256_setzero_ps(); }
        const std::uint8_t* code_row = m.codes + static_cast<std::size_t>(row) * (m.columns / 2);
        for (int g = 0; g < groups; ++g) {
            const float scale = e4m3[m.scales[scale_offset(row, g, m.columns)]] * inverse;
            // 8 code bytes -> 16 nibbles in column order (low nibble first).
            std::uint64_t packed = 0;
            std::memcpy(&packed, code_row + g * (kGroup / 2), 8);
            const __m128i bytes = _mm_cvtsi64_si128(static_cast<long long>(packed));
            const __m128i lo    = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
            const __m128i hi    = _mm_and_si128(_mm_srli_epi16(bytes, 4), _mm_set1_epi8(0x0F));
            const __m128i nib   = _mm_unpacklo_epi8(lo, hi);
            __m256 w[2];
            for (int half = 0; half < 2; ++half) {
                const __m256i idx  = _mm256_cvtepu8_epi32(half == 0 ? nib : _mm_srli_si128(nib, 8));
                const __m256 mag   = _mm256_permutevar8x32_ps(lut, _mm256_and_si256(idx, low7));
                const __m256i sign = _mm256_slli_epi32(_mm256_and_si256(idx, sign8), 28);
                w[half] = _mm256_mul_ps(_mm256_xor_ps(mag, _mm256_castsi256_ps(sign)), _mm256_set1_ps(scale));
            }
            const int k0 = g * kGroup;
            for (int c = 0; c < ncols; ++c) {
                acc[c] = _mm256_fmadd_ps(w[0], _mm256_loadu_ps(x[c] + k0), acc[c]);
                acc[c] = _mm256_fmadd_ps(w[1], _mm256_loadu_ps(x[c] + k0 + 8), acc[c]);
            }
        }
        for (int c = 0; c < ncols; ++c) {
            const __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc[c]), _mm256_extractf128_ps(acc[c], 1));
            const __m128 t = _mm_add_ps(s, _mm_movehl_ps(s, s));
            out[static_cast<std::size_t>(c) * out_stride + row - out_row0] =
                _mm_cvtss_f32(_mm_add_ss(t, _mm_movehdup_ps(t)));
        }
    }
}

} // namespace

float flash_next_nvfp4_weight(const HostNvfp4Matrix& matrix, std::int32_t row, std::int32_t column) {
    const std::uint8_t byte = matrix.codes[static_cast<std::size_t>(row) * (matrix.columns / 2) + column / 2];
    const int code          = (column & 1) ? (byte >> 4) : (byte & 0x0F);
    const float magnitude   = kE2m1[static_cast<std::size_t>(code & 7)];
    const float value       = (code & 8) ? -magnitude : magnitude;
    const float scale       = e4m3_table()[matrix.scales[scale_offset(row, column / kGroup, matrix.columns)]];
    return value * scale / matrix.divisor;
}

void flash_next_cold_experts(std::span<const FlashNextColdExpert> work, const std::uint16_t* x_bf16,
                             std::int32_t tokens, float* out, ops::cpu::SpinPool& pool) {
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
        throw std::runtime_error("Flash-Next host experts need AVX2 and FMA");
    }
    std::fill(out, out + static_cast<std::size_t>(tokens) * kHidden, 0.0F);
    if (work.empty()) { return; }
    std::vector<float> x(static_cast<std::size_t>(tokens) * kHidden);
    for (std::size_t i = 0; i < x.size(); ++i) { x[i] = bf16_to_float(x_bf16[i]); }
    std::vector<std::size_t> base(work.size() + 1, 0);
    for (std::size_t w = 0; w < work.size(); ++w) {
        const auto& e = work[w];
        if (e.columns.size() != e.weights.size() || e.columns.empty() || e.gate_up.rows != 2 * kIntermediate ||
            e.gate_up.columns != kHidden || e.down.rows != kHidden || e.down.columns != kIntermediate) {
            throw std::invalid_argument("Flash-Next host experts: invalid work item");
        }
        base[w + 1] = base[w] + e.columns.size();
    }
    std::vector<float> gate_up(base.back() * 2 * kIntermediate);
    std::vector<float> act(base.back() * kIntermediate);

    constexpr int kGateBlocks = 2 * kIntermediate / kRowBlock;
    pool.run(work.size() * kGateBlocks, [&](std::size_t task) {
        const std::size_t w = task / kGateBlocks;
        const int block     = static_cast<int>(task % kGateBlocks);
        const auto& e       = work[w];
        for (std::size_t c0 = 0; c0 < e.columns.size(); c0 += kMaxColumns) {
            const int n = static_cast<int>(std::min<std::size_t>(kMaxColumns, e.columns.size() - c0));
            const float* xs[kMaxColumns];
            for (int c = 0; c < n; ++c) { xs[c] = x.data() + static_cast<std::size_t>(e.columns[c0 + c]) * kHidden; }
            rows_avx2(e.gate_up, block * kRowBlock, (block + 1) * kRowBlock, xs, n,
                      gate_up.data() + (base[w] + c0) * 2 * kIntermediate, 2 * kIntermediate, 0);
        }
    });
    for (std::size_t i = 0; i < base.back(); ++i) {
        const float* gu = gate_up.data() + i * 2 * kIntermediate;
        float* a        = act.data() + i * kIntermediate;
        for (int j = 0; j < kIntermediate; ++j) {
            const float g = gu[j];
            a[j]          = round_bf16(g / (1.0F + std::exp(-g)) * gu[kIntermediate + j]);
        }
    }
    constexpr int kDownBlocks = kHidden / kRowBlock;
    pool.run(kDownBlocks, [&](std::size_t block) {
        const int row_begin = static_cast<int>(block) * kRowBlock;
        float partial[kMaxColumns * kRowBlock];
        for (std::size_t w = 0; w < work.size(); ++w) {
            const auto& e = work[w];
            for (std::size_t c0 = 0; c0 < e.columns.size(); c0 += kMaxColumns) {
                const int n = static_cast<int>(std::min<std::size_t>(kMaxColumns, e.columns.size() - c0));
                const float* as[kMaxColumns];
                for (int c = 0; c < n; ++c) { as[c] = act.data() + (base[w] + c0 + c) * kIntermediate; }
                rows_avx2(e.down, row_begin, row_begin + kRowBlock, as, n, partial, kRowBlock, row_begin);
                for (int c = 0; c < n; ++c) {
                    float* o           = out + static_cast<std::size_t>(e.columns[c0 + c]) * kHidden + row_begin;
                    const float weight = e.weights[c0 + c];
                    for (int r = 0; r < kRowBlock; ++r) { o[r] += weight * partial[c * kRowBlock + r]; }
                }
            }
        }
    });
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
