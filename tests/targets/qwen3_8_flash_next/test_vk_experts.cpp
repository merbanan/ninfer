// Flash-Next RX 570 (Vulkan) expert tier: shader correctness against the CPU oracle
// (flash_next_cold_experts, the same one the host cold path uses) on synthetic NVFP4 experts, and
// a rough per-call overhead measurement. Skips (exit 77) when no AMD Vulkan device is present, so
// it runs only where NINFER_VULKAN_EXPERTS was built and an RX 570 (or another AMD GPU) is here.
#include "artifact/reader.h"
#include "ops/sparse_moe/cpu/sparse_moe_cpu.h"
#include "targets/qwen3_8_flash_next/impl/expert_bank.h"
#include "targets/qwen3_8_flash_next/impl/expert_cpu.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"
#include "targets/qwen3_8_flash_next/impl/vk/vk_experts.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace ninfer::targets::qwen3_8_flash_next::detail;

namespace {

constexpr int kHidden = 2560;
constexpr int kInter  = 640;

std::uint16_t to_bf16(float v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, 4);
    return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
}

std::size_t align256(std::size_t v) { return (v + 255U) & ~std::size_t{255U}; }

struct HostBank {
    std::vector<std::byte> bytes;
};

HostBank random_bank(std::mt19937& rng, int experts, int rows, int columns, float divisor) {
    const auto g = ninfer::artifact::block_scale_bank_geometry(
        ninfer::artifact::NumericFormat::NVFP4,
        std::array<std::uint64_t, 3>{static_cast<std::uint64_t>(experts), static_cast<std::uint64_t>(rows),
                                     static_cast<std::uint64_t>(columns)});
    HostBank bank;
    bank.bytes.assign(g.encoded_bytes, std::byte{0});
    auto* p = reinterpret_cast<std::uint8_t*>(bank.bytes.data());
    for (std::uint64_t i = 0; i < g.code_plane_bytes; i += 4) {
        const std::uint32_t r = rng();
        std::memcpy(p + i, &r, 4);
    }
    for (std::uint64_t i = 0; i < g.scale_plane_bytes; ++i) {
        p[g.scale_plane_offset + i] = static_cast<std::uint8_t>(((3 + rng() % 6) << 3) | (rng() & 7));
    }
    auto* div = reinterpret_cast<float*>(p + g.weight_divisor_offset);
    for (int e = 0; e < experts; ++e) { div[e] = divisor * (1.0F + 0.01F * static_cast<float>(e % 7)); }
    return bank;
}

// Same byte layout as FlashNextExpertCache::pack()/packed_bytes() (expert_cache.cpp): gate codes,
// gate scales, gate divisor, down codes, down scales, down divisor, each 256-byte aligned.
void pack_expert(const Nvfp4ExpertBankView& gate, const Nvfp4ExpertBankView& down, std::int32_t expert,
                 std::byte* dst) {
    std::size_t at = 0;
    const auto put = [&](const std::byte* src, std::size_t bytes) {
        std::memcpy(dst + at, src, bytes);
        at += align256(bytes);
    };
    put(gate.codes + static_cast<std::size_t>(expert) * gate.code_bytes_per_expert, gate.code_bytes_per_expert);
    put(gate.scales + static_cast<std::size_t>(expert) * gate.scale_bytes_per_expert, gate.scale_bytes_per_expert);
    put(reinterpret_cast<const std::byte*>(gate.weight_scale_divisors + expert), sizeof(float));
    put(down.codes + static_cast<std::size_t>(expert) * down.code_bytes_per_expert, down.code_bytes_per_expert);
    put(down.scales + static_cast<std::size_t>(expert) * down.scale_bytes_per_expert, down.scale_bytes_per_expert);
    put(reinterpret_cast<const std::byte*>(down.weight_scale_divisors + expert), sizeof(float));
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

int main() {
    constexpr int kBankExperts = 8;
    std::mt19937 rng(2026);

    HostBank gate_bank = random_bank(rng, kBankExperts, 2 * kInter, kHidden, 96.0F);
    HostBank down_bank = random_bank(rng, kBankExperts, kHidden, kInter, 256.0F);
    MoeWeights weights{};
    weights.expert_gate_up = make_nvfp4_expert_bank_view(gate_bank.bytes.data(), gate_bank.bytes.size(), kBankExperts,
                                                         2 * kInter, kHidden);
    weights.expert_down =
        make_nvfp4_expert_bank_view(down_bank.bytes.data(), down_bank.bytes.size(), kBankExperts, kHidden, kInter);

    FlashNextVkPolicy policy;
    policy.cache_bytes = 64ULL << 20; // comfortably holds kBankExperts small synthetic experts
    policy.promote_per_call = kBankExperts;
    auto vk = FlashNextVkExperts::create(weights, policy);
    if (!vk) {
        std::printf("SKIP: no usable AMD Vulkan device\n");
        return 77;
    }

    // Promote every synthetic expert (key == expert index; layer is irrelevant here) and wait for
    // residency. There are fewer promotion staging buffers than kBankExperts, so a begin_promotion
    // for an expert with no free buffer is a no-op; retry each not-yet-resident, not-yet-in-flight
    // expert every spin, same as expert_cache.cpp does across calls.
    std::vector<std::byte> packed(vk->packed_bytes());
    for (int spins = 0; spins < 100000; ++spins) {
        vk->retire_promotions();
        bool all_resident = true;
        for (int e = 0; e < kBankExperts; ++e) {
            if (vk->resident(e)) { continue; }
            all_resident = false;
            if (!vk->promotion_in_flight(e)) {
                pack_expert(weights.expert_gate_up, weights.expert_down, e, packed.data());
                vk->begin_promotion(e, packed.data());
            }
        }
        if (all_resident) { break; }
    }
    for (int e = 0; e < kBankExperts; ++e) {
        if (!vk->resident(e)) {
            std::printf("FAIL: expert %d never became resident\n", e);
            return 1;
        }
    }

    ninfer::ops::cpu::SpinPool pool(4);
    int failures = 0;
    std::normal_distribution<float> normal(0.0F, 1.0F);

    const auto run = [&](int tokens, int jobs_count, const char* label) {
        std::vector<std::uint16_t> x(static_cast<std::size_t>(tokens) * kHidden);
        for (auto& v : x) { v = to_bf16(normal(rng)); }

        std::vector<FlashNextVkJob> jobs(jobs_count);
        std::vector<FlashNextColdExpert> work(jobs_count);
        for (int j = 0; j < jobs_count; ++j) {
            const int expert = j % kBankExperts;
            jobs[j].slot      = vk->slot_of(expert);
            work[j].gate_up   = host_matrix(weights.expert_gate_up, expert);
            work[j].down      = host_matrix(weights.expert_down, expert);
            // Overlapping token columns across jobs, like several of a call's 10 routed experts
            // sharing tokens.
            for (int t = 0; t < tokens; ++t) {
                if (((t + j) % 3) != 0) { continue; }
                jobs[j].columns.push_back(t);
                jobs[j].weights.push_back(0.05F + 0.01F * static_cast<float>(t + j));
            }
            work[j].columns = jobs[j].columns;
            work[j].weights = jobs[j].weights;
        }
        // Drop jobs that ended up empty (no columns landed in them); both lists must stay non-empty.
        std::vector<FlashNextVkJob> nonempty_jobs;
        std::vector<FlashNextColdExpert> nonempty_work;
        for (int j = 0; j < jobs_count; ++j) {
            if (!jobs[j].columns.empty()) {
                nonempty_jobs.push_back(jobs[j]);
                nonempty_work.push_back(work[j]);
            }
        }

        std::vector<float> ref(static_cast<std::size_t>(tokens) * kHidden, 0.0F);
        flash_next_cold_experts(nonempty_work, x.data(), tokens, ref.data(), pool);

        std::vector<float> got(static_cast<std::size_t>(tokens) * kHidden, 0.0F);
        const auto start = std::chrono::steady_clock::now();
        vk->submit(nonempty_jobs, x.data(), tokens);
        vk->wait(got.data(), tokens);
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();

        double max_ref = 0.0, max_err = 0.0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            max_ref = std::max(max_ref, std::abs(static_cast<double>(ref[i])));
            max_err = std::max(max_err, std::abs(static_cast<double>(ref[i] - got[i])));
        }
        const bool ok = std::isfinite(max_err) && max_ref > 0.0 && max_err <= 2e-2 * max_ref;
        std::printf("%-24s T=%-2d jobs=%-2zu max|ref| %.4g max err %.3g %.1f us %s\n", label, tokens,
                    nonempty_jobs.size(), max_ref, max_err, us, ok ? "ok" : "FAIL");
        failures += ok ? 0 : 1;
    };

    run(1, 1, "single expert");
    run(1, 5, "T=1 overhead x5");
    run(1, kBankExperts, "T=1 overhead x8");
    run(3, 5, "shared columns");
    run(8, kBankExperts, "full decode width");

    vk->report();
    std::printf(failures == 0 ? "PASS\n" : "FAIL\n");
    return failures == 0 ? 0 : 1;
}
