// Flash-Next host-resident experts through the device expert cache against the same banks fully
// device resident, on synthetic weights: decode calls with host-computed misses, promotions,
// hybrid and uploading prefill, and cached hits after a prefill.
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/expert_cache.h"
#include "targets/qwen3_8_flash_next/impl/moe.h"
#include "targets/qwen3_8_flash_next/impl/moe_kernels.h"
#include "targets/qwen3_8_flash_next/impl/moe_prefill_turing.h"
#include "targets/qwen3_8_flash_next/impl/moe_route.h"
#include "targets/qwen3_8_flash_next/impl/moe_workspace.h"
#include "targets/qwen3_8_flash_next/impl/stage_ledger.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::targets::qwen3_8_flash_next::detail;

namespace {

constexpr int kHidden = 2560;
constexpr int kInter  = 640;

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

struct HostBank {
    std::vector<std::byte> bytes;
    int rows = 0, columns = 0;
};

HostBank random_bank(std::mt19937& rng, int rows, int columns, float divisor) {
    const auto g = artifact::block_scale_bank_geometry(
        artifact::NumericFormat::NVFP4,
        std::array<std::uint64_t, 3>{512, static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(columns)});
    HostBank bank;
    bank.rows    = rows;
    bank.columns = columns;
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
    for (int e = 0; e < 512; ++e) { div[e] = divisor * (1.0F + 0.01F * static_cast<float>(e % 7)); }
    return bank;
}

struct DeviceWeight {
    DeviceBuffer buffer;
    Weight weight;
};

DeviceWeight bf16_weight(std::mt19937& rng, int rows, int columns, float scale) {
    std::vector<std::uint16_t> host(static_cast<std::size_t>(rows) * columns);
    std::normal_distribution<float> normal(0.0F, scale);
    for (auto& v : host) { v = to_bf16(normal(rng)); }
    DeviceWeight out{DeviceBuffer(host.size() * 2), Weight{}};
    out.buffer.copy_from_host(host.data(), host.size() * 2);
    Weight& w          = out.weight;
    w.payload          = out.buffer.p;
    w.payload_bytes    = host.size() * 2;
    w.qdata            = out.buffer.p;
    w.qtype            = QType::BF16_CTRL;
    w.layout           = QuantLayout::Contiguous;
    w.n                = rows;
    w.k                = columns;
    w.ndim             = 2;
    w.shape[0]         = rows;
    w.shape[1]         = columns;
    w.padded_shape[0]  = rows;
    w.padded_shape[1]  = columns;
    return out;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    setenv("NINFER_FLASH_NEXT_CACHE_MB", "1500", 0);
    setenv("NINFER_FLASH_NEXT_ADMIT", "1", 0);
    setenv("NINFER_FLASH_NEXT_PROMOTE", "4", 0);
    setenv("NINFER_FLASH_NEXT_HYBRID", "48", 0);
    std::mt19937 rng(11);

    HostBank gate = random_bank(rng, 2 * kInter, kHidden, 96.0F);
    HostBank down = random_bank(rng, kHidden, kInter, 256.0F);
    DeviceBuffer gate_dev(gate.bytes.size());
    DeviceBuffer down_dev(down.bytes.size());
    gate_dev.copy_from_host(gate.bytes.data(), gate.bytes.size());
    down_dev.copy_from_host(down.bytes.data(), down.bytes.size());

    auto router      = bf16_weight(rng, 512, kHidden, 0.05F);
    auto shared_down = bf16_weight(rng, kHidden, kInter, 0.02F);
    auto shared_gate = bf16_weight(rng, kInter, kHidden, 0.02F);
    auto shared_up   = bf16_weight(rng, kInter, kHidden, 0.02F);
    auto shared_gw   = bf16_weight(rng, 1, kHidden, 0.02F);

    MoeWeights device_weights{router.weight, shared_down.weight, shared_gate.weight, shared_up.weight,
                              shared_gw.weight,
                              make_nvfp4_expert_bank_view(gate_dev.p, gate_dev.bytes, 512, 2 * kInter, kHidden),
                              make_nvfp4_expert_bank_view(down_dev.p, down_dev.bytes, 512, kHidden, kInter)};
    MoeWeights host_weights = device_weights;
    host_weights.expert_gate_up =
        make_nvfp4_expert_bank_view(gate.bytes.data(), gate.bytes.size(), 512, 2 * kInter, kHidden);
    host_weights.expert_down = make_nvfp4_expert_bank_view(down.bytes.data(), down.bytes.size(), 512, kHidden, kInter);
    for (auto* bank : {&host_weights.expert_gate_up, &host_weights.expert_down}) {
        bank->mapped_host          = true;
        bank->mapped_payload       = bank->codes;
        bank->mapped_payload_bytes = 1;
    }

    constexpr int kMaxTokens = 4096;
    DeviceArena workspace(flash_next_moe_workspace_capacity_bytes(1, kMaxTokens) + (1U << 20));
    DeviceArena io((static_cast<std::size_t>(kHidden) * kMaxTokens * 2 + 4096) * 3);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::normal_distribution<float> normal(0.0F, 1.0F);
    int failures = 0;
    const auto run = [&](const std::vector<std::uint16_t>& x, int tokens, const char* label, bool exact) {
        io.reset();
        Tensor input     = io.alloc(DType::BF16, {kHidden, tokens});
        Tensor reference = io.alloc(DType::BF16, {kHidden, tokens});
        Tensor cached    = io.alloc(DType::BF16, {kHidden, tokens});
        CUDA_CHECK(cudaMemcpy(input.data, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
        flash_next_moe(input, device_weights, reference, workspace, stream);
        flash_next_moe(input, host_weights, cached, workspace, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<std::uint16_t> a(x.size()), b(x.size());
        CUDA_CHECK(cudaMemcpy(a.data(), reference.data, a.size() * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), cached.data, b.size() * 2, cudaMemcpyDeviceToHost));
        double max_ref = 0.0, max_err = 0.0;
        std::size_t differ = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double r = from_bf16(a[i]);
            const double c = from_bf16(b[i]);
            max_ref        = std::max(max_ref, std::abs(r));
            max_err        = std::max(max_err, std::abs(r - c));
            differ += a[i] != b[i];
        }
        const auto& st = flash_next_expert_cache()->stats();
        const bool ok  = std::isfinite(max_err) && max_ref > 0.0 && (exact ? differ == 0 : max_err <= 2e-2 * max_ref);
        std::printf("%-26s T=%-3d max|ref| %.4g max err %.3g differing %zu | hits %llu/%llu cold %llu promo %llu "
                    "uploads %llu hybrid %llu %s\n",
                    label, tokens, max_ref, max_err, differ, static_cast<unsigned long long>(st.hits),
                    static_cast<unsigned long long>(st.lookups), static_cast<unsigned long long>(st.cold),
                    static_cast<unsigned long long>(st.promotions), static_cast<unsigned long long>(st.uploads),
                    static_cast<unsigned long long>(st.hybrid_calls), ok ? "ok" : "FAIL");
        failures += ok ? 0 : 1;
    };
    const auto random_x = [&](int tokens) {
        std::vector<std::uint16_t> x(static_cast<std::size_t>(tokens) * kHidden);
        for (auto& v : x) { v = to_bf16(normal(rng)); }
        return x;
    };

    // Decode: first call all misses (host), the same token again after promotions settle.
    const auto x1 = random_x(1);
    run(x1, 1, "decode cold", false);
    CUDA_CHECK(cudaDeviceSynchronize());
    run(x1, 1, "decode after promotions", false);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < 3; ++i) { run(x1, 1, "decode repeat", false); CUDA_CHECK(cudaDeviceSynchronize()); }
    run(x1, 1, "decode all cached", true);
    run(random_x(3), 3, "decode 3 tokens", false);
    // Prefill: staged banks, uploads kept in the cache; then a decode of one of its tokens hits.
    // Hybrid prefill: experts with few columns on the host, the rest staged.
    const auto x40 = random_x(40);
    run(x40, 40, "prefill (hybrid)", false);
    run(x40, 40, "prefill again", false);
    std::vector<std::uint16_t> col(x40.begin() + 7 * kHidden, x40.begin() + 8 * kHidden);
    run(col, 1, "decode prefill token", false);
    // Large prefill: every missing expert uploaded (above the hybrid bound), then kept cached.
    const auto x64 = random_x(64);
    run(x64, 64, "prefill (uploads)", true);
    run(x64, 64, "prefill again (cached)", true);
    // Optional timing of the device-resident MoE (NINFER_TEST_TIME_MOE=1).
    if (const char* t = std::getenv("NINFER_TEST_TIME_MOE"); t != nullptr && t[0] == '1') {
        for (int tokens : {1, 16, 64, 256, 1024, 4096}) {
            io.reset();
            const auto x = random_x(tokens);
            Tensor input  = io.alloc(DType::BF16, {kHidden, tokens});
            Tensor out    = io.alloc(DType::BF16, {kHidden, tokens});
            CUDA_CHECK(cudaMemcpy(input.data, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
            cudaEvent_t a, b;
            CUDA_CHECK(cudaEventCreate(&a));
            CUDA_CHECK(cudaEventCreate(&b));
            flash_next_moe(input, device_weights, out, workspace, stream);
            CUDA_CHECK(cudaEventRecord(a, stream));
            for (int i = 0; i < 10; ++i) { flash_next_moe(input, device_weights, out, workspace, stream); }
            CUDA_CHECK(cudaEventRecord(b, stream));
            CUDA_CHECK(cudaEventSynchronize(b));
            float ms = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
            std::printf("device MoE T=%d: %.3f ms/call, %.1f us/token\n", tokens, ms / 10, ms * 100 / tokens);
            const auto scope = workspace.scope();
            FlashNextMoeWorkspace scratch = allocate_flash_next_moe_workspace(workspace, tokens);
            const auto time = [&](const char* what, auto&& body) {
                body();
                CUDA_CHECK(cudaEventRecord(a, stream));
                for (int i = 0; i < 10; ++i) { body(); }
                CUDA_CHECK(cudaEventRecord(b, stream));
                CUDA_CHECK(cudaEventSynchronize(b));
                float t = 0.0F;
                CUDA_CHECK(cudaEventElapsedTime(&t, a, b));
                std::printf("  %-10s %.1f us\n", what, t * 100);
            };
            time("route", [&] {
                flash_next_route(input, device_weights.router, device_weights.shared_gate_weight, scratch.scores,
                                 scratch.ids, scratch.alpha, scratch.shared_scale, stream);
            });
            time("experts", [&] { flash_next_moe_kernels_launch(input, device_weights, scratch, out, stream); });
            if (tokens > 8) {
                // Turing tensor-core prefill against the SIMT kernels on the same routing.
                std::vector<std::uint16_t> simt(x.size()), turing(x.size());
                flash_next_moe_prefill_turing_override(false);
                flash_next_moe_kernels_launch(input, device_weights, scratch, out, stream);
                CUDA_CHECK(cudaMemcpyAsync(simt.data(), out.data, simt.size() * 2, cudaMemcpyDeviceToHost, stream));
                time("simt", [&] { flash_next_moe_kernels_launch(input, device_weights, scratch, out, stream); });
                flash_next_moe_prefill_turing_override(true);
                flash_next_moe_kernels_launch(input, device_weights, scratch, out, stream);
                CUDA_CHECK(cudaMemcpyAsync(turing.data(), out.data, turing.size() * 2, cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                time("turing", [&] { flash_next_moe_kernels_launch(input, device_weights, scratch, out, stream); });
                if (FlashNextStageLedger::is_enabled()) {
                    FlashNextStageLedger::instance().begin_chunk(stream, tokens);
                    flash_next_route(input, device_weights.router, device_weights.shared_gate_weight, scratch.scores,
                                     scratch.ids, scratch.alpha, scratch.shared_scale, stream);
                    stage_ledger_record(stream, FlashNextStageId::MoE_Router);
                    flash_next_moe_kernels_launch(input, device_weights, scratch, out, stream);
                    FlashNextStageLedger::instance().finish_chunk(stream);
                }
                double max_ref = 0.0, max_err = 0.0, sum_err = 0.0;
                for (std::size_t i = 0; i < simt.size(); ++i) {
                    const double r = from_bf16(simt[i]);
                    max_ref        = std::max(max_ref, std::abs(r));
                    max_err        = std::max(max_err, std::abs(r - from_bf16(turing[i])));
                    sum_err += std::abs(r - from_bf16(turing[i]));
                }
                const bool ok = std::isfinite(max_err) && max_err <= 2e-2 * max_ref;
                std::printf("  turing vs simt: max|ref| %.4g max err %.3g mean err %.3g %s\n", max_ref, max_err,
                            sum_err / static_cast<double>(simt.size()), ok ? "ok" : "FAIL");
                failures += ok ? 0 : 1;
            }
            if (tokens <= 8) {
                time("down", [&] {
                    flash_next_moe_down_launch(flash_next_moe_down_kernel_selection(), device_weights, scratch, tokens,
                                               out, stream);
                });
            }
        }
    }
    flash_next_expert_cache()->report();
    std::printf(failures == 0 ? "PASS\n" : "FAIL\n");
    return failures == 0 ? 0 : 1;
}
