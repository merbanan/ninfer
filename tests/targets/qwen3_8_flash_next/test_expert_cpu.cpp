#include "targets/qwen3_8_flash_next/impl/expert_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace fn = ninfer::targets::qwen3_8_flash_next::detail;

namespace {

struct Matrix {
    std::vector<std::uint8_t> codes;
    std::vector<std::uint8_t> scales;
    fn::HostNvfp4Matrix view;
};

Matrix random_matrix(std::mt19937& rng, int rows, int columns, float divisor) {
    Matrix m;
    m.codes.resize(static_cast<std::size_t>(rows) * columns / 2);
    m.scales.resize(static_cast<std::size_t>(rows) * columns / 16);
    for (auto& b : m.codes) { b = static_cast<std::uint8_t>(rng()); }
    // E4M3 scales in [2^-4, 2^2): exponent field 3..8, any mantissa, positive.
    for (auto& s : m.scales) { s = static_cast<std::uint8_t>(((3 + rng() % 6) << 3) | (rng() & 7)); }
    m.view = {m.codes.data(), m.scales.data(), divisor, rows, columns};
    return m;
}

std::uint16_t to_bf16(float v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, 4);
    return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
}

float from_bf16(std::uint16_t h) {
    const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float v                  = 0.0F;
    std::memcpy(&v, &bits, 4);
    return v;
}

float round_bf16(float v) { return from_bf16(to_bf16(v)); }

} // namespace

int main() {
    constexpr int kHidden = 2560;
    constexpr int kInter  = 640;
    constexpr int kTokens = 21;
    std::mt19937 rng(7);

    std::vector<Matrix> gate_up;
    std::vector<Matrix> down;
    for (int e = 0; e < 3; ++e) {
        gate_up.push_back(random_matrix(rng, 2 * kInter, kHidden, 64.0F + e));
        down.push_back(random_matrix(rng, kHidden, kInter, 128.0F));
    }
    std::vector<std::uint16_t> x(static_cast<std::size_t>(kTokens) * kHidden);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (auto& v : x) { v = to_bf16(normal(rng)); }

    std::vector<fn::FlashNextColdExpert> work(3);
    for (int e = 0; e < 3; ++e) {
        work[e].gate_up = gate_up[e].view;
        work[e].down    = down[e].view;
    }
    // Expert 0: one column (decode shape); expert 1: 18 columns (crosses the 16-column tile);
    // expert 2: a few columns shared with expert 1.
    work[0].columns = {5};
    for (int c = 0; c < 18; ++c) { work[1].columns.push_back(c + 2); }
    work[2].columns = {0, 5, 20};
    for (auto& w : work) {
        for (std::size_t i = 0; i < w.columns.size(); ++i) { w.weights.push_back(0.05F + 0.01F * float(i)); }
    }

    ninfer::ops::cpu::SpinPool pool(4);
    std::vector<float> out(static_cast<std::size_t>(kTokens) * kHidden, 123.0F);
    fn::flash_next_cold_experts(work, x.data(), kTokens, out.data(), pool);

    std::vector<double> ref(out.size(), 0.0);
    for (const auto& w : work) {
        for (std::size_t i = 0; i < w.columns.size(); ++i) {
            const int col = w.columns[i];
            std::vector<float> act(kInter);
            for (int j = 0; j < kInter; ++j) {
                double g = 0.0;
                double u = 0.0;
                for (int k = 0; k < kHidden; ++k) {
                    const double xv = from_bf16(x[static_cast<std::size_t>(col) * kHidden + k]);
                    g += fn::flash_next_nvfp4_weight(w.gate_up, j, k) * xv;
                    u += fn::flash_next_nvfp4_weight(w.gate_up, kInter + j, k) * xv;
                }
                act[j] = round_bf16(static_cast<float>(g / (1.0 + std::exp(-g)) * u));
            }
            for (int r = 0; r < kHidden; ++r) {
                double d = 0.0;
                for (int j = 0; j < kInter; ++j) { d += fn::flash_next_nvfp4_weight(w.down, r, j) * act[j]; }
                ref[static_cast<std::size_t>(col) * kHidden + r] += w.weights[i] * d;
            }
        }
    }
    double max_err = 0.0;
    double max_ref = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        max_err = std::max(max_err, std::abs(out[i] - ref[i]));
        max_ref = std::max(max_ref, std::abs(ref[i]));
    }
    std::printf("max |ref| %.4g, max err %.4g\n", max_ref, max_err);
    // BF16 activation rounding can flip on FP32 vs FP64 sums: allow a small relative slack.
    if (!(max_ref > 0.0) || max_err > 2e-2 * max_ref) {
        std::printf("FAIL\n");
        return 1;
    }
    for (int r = 0; r < kHidden; ++r) {
        if (out[static_cast<std::size_t>(1) * kHidden + r] != 0.0F) {
            std::printf("FAIL: unrouted column not zero\n");
            return 1;
        }
    }
    std::printf("PASS\n");
    return 0;
}
