#include "targets/qwen3_8_flash_next/impl/moe_prefill_turing.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/turing_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

constexpr int kHidden       = 2560;
constexpr int kIntermediate = 640;
constexpr int kTopK         = 10;
[[maybe_unused]] constexpr int kPaths = kTopK + 1;
constexpr int kThreads      = 128; // four warps
constexpr int kRows         = 128; // weight rows per CTA (32 per warp)
[[maybe_unused]] constexpr int kTokens = 32; // routed columns per tile
constexpr int kKStep        = 64;  // one scale tile: four K16 groups
[[maybe_unused]] constexpr int kStride = kKStep + 8;
constexpr int kGridY        = 96;

// Global data of one K step for this thread: 32 code bytes and 4 scale bytes of weight row
// `threadIdx.x`, and 16 BF16 activations of token row threadIdx.x / 4.
struct StepRegs {
    uint4 codes[2];
    std::uint32_t scales;
    uint4 x[2];
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750

__device__ __forceinline__ __half2 e2m1x2_to_half2(std::uint32_t byte) {
    // Exact: the E2M1 bits at FP16 bits 11..9 are the value times 2^-14 (see nvfp4_codec.cuh).
    const std::uint32_t bits = ((byte & 0x07U) << 9) | ((byte & 0x08U) << 12) | ((byte & 0x70U) << 21) |
                               ((byte & 0x80U) << 24);
    __half2_raw raw;
    raw.x = static_cast<unsigned short>(bits);
    raw.y = static_cast<unsigned short>(bits >> 16);
    return __hmul2(__half2(raw), __half2half2(__ushort_as_half(0x7400U))); // 2^14
}

__device__ __forceinline__ __half e4m3_to_half(std::uint32_t byte) {
    __half_raw raw;
    raw.x = static_cast<unsigned short>(((byte & 0x7FU) << 7) | ((byte & 0x80U) << 8));
    return __hmul(__half(raw), __ushort_as_half(0x5C00U)); // 2^8
}

__device__ __forceinline__ std::uint32_t half2_bits(__half2 value) {
    return *reinterpret_cast<std::uint32_t*>(&value);
}

__device__ __forceinline__ std::uint32_t bf16x2_to_half2_bits(std::uint32_t bits) {
    const float lo = __uint_as_float(bits << 16);
    const float hi = __uint_as_float(bits & 0xFFFF0000U);
    constexpr float kMax = 65504.0F;
    return half2_bits(__floats2half2_rn(fminf(fmaxf(lo, -kMax), kMax), fminf(fmaxf(hi, -kMax), kMax)));
}

// Weight row `local` of the CTA -> row of the expert matrix.
template <bool GateUp>
__device__ __forceinline__ int weight_row(int tile, int local) {
    if constexpr (GateUp) {
        const int pair = tile * (kRows / 2) + (local & 63);
        return local < 64 ? pair : kIntermediate + pair;
    } else {
        return tile * kRows + local;
    }
}

#endif

template <bool GateUp>
__global__ __launch_bounds__(kThreads) void flash_next_moe_prefill_turing_kernel(
    FlashNextTuringGroups groups, FlashNextTuringBank bank, const __nv_bfloat16* __restrict__ x_source,
    __nv_bfloat16* __restrict__ activations, float* __restrict__ down_intermediate) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
    constexpr int kK      = GateUp ? kHidden : kIntermediate;
    constexpr int kSteps  = kK / kKStep;
    constexpr int kKTiles = kK / 64;
    __shared__ __align__(16) __half x_sh[kTokens * kStride];
    __shared__ __align__(16) __half w_sh[kRows * kStride];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int tile = static_cast<int>(blockIdx.x);
    const int row  = weight_row<GateUp>(tile, tid);
    const int x_row = tid >> 2;
    const int x_col = (tid & 3) * 16;
    const int active = groups.active_count[0];

    for (int act = static_cast<int>(blockIdx.y); act < active; act += static_cast<int>(gridDim.y)) {
        const int expert = groups.active_experts[act];
        const int count  = groups.expert_counts[expert];
        const int start  = groups.expert_offsets[expert];
        const int bank_index = groups.expert_slots != nullptr ? groups.expert_slots[expert] : expert;
        const std::uint8_t* codes = bank.codes + static_cast<std::uint64_t>(bank_index) * bank.code_stride +
                                    static_cast<std::int64_t>(row) * (kK / 2);
        const std::uint8_t* scales = bank.scales + static_cast<std::uint64_t>(bank_index) * bank.scale_stride +
                                     static_cast<std::int64_t>(row / 128) * kKTiles * 512 + (row & 31) * 16 +
                                     ((row & 127) >> 5) * 4;
        const float inverse = 1.0F / bank.divisors[bank_index];

        for (int base = 0; base < count; base += kTokens) {
            const int valid = min(kTokens, count - base);
            const __nv_bfloat16* x_ptr = nullptr;
            if (x_row < valid) {
                const int pos   = start + base + x_row;
                const int token = groups.grouped_tokens[pos];
                x_ptr = GateUp ? x_source + static_cast<std::int64_t>(token) * kHidden
                               : x_source + (static_cast<std::int64_t>(token) * kPaths + groups.grouped_paths[pos]) *
                                                kIntermediate;
                x_ptr += x_col;
            }
            const auto load = [&](int step, StepRegs& r) {
                const auto* c = reinterpret_cast<const uint4*>(codes + step * (kKStep / 2));
                r.codes[0]    = c[0];
                r.codes[1]    = c[1];
                r.scales      = *reinterpret_cast<const std::uint32_t*>(scales + step * 512);
                if (x_ptr != nullptr) {
                    const auto* xs = reinterpret_cast<const uint4*>(x_ptr + step * kKStep);
                    r.x[0]         = xs[0];
                    r.x[1]         = xs[1];
                } else {
                    r.x[0] = make_uint4(0, 0, 0, 0);
                    r.x[1] = make_uint4(0, 0, 0, 0);
                }
            };
            float d[4][2][4] = {};
            StepRegs regs;
            load(0, regs);
            for (int step = 0; step < kSteps; ++step) {
                __syncthreads();
                // Decode this thread's weight row (4 groups x 16 values) and token chunk to FP16.
                {
                    const std::uint32_t words[8] = {regs.codes[0].x, regs.codes[0].y, regs.codes[0].z,
                                                    regs.codes[0].w, regs.codes[1].x, regs.codes[1].y,
                                                    regs.codes[1].z, regs.codes[1].w};
                    auto* dst = reinterpret_cast<uint4*>(w_sh + tid * kStride);
#pragma unroll
                    for (int group = 0; group < 4; ++group) {
                        const __half2 scale = __half2half2(e4m3_to_half(regs.scales >> (8 * group)));
                        std::uint32_t out[8];
#pragma unroll
                        for (int b = 0; b < 8; ++b) {
                            const std::uint32_t byte = words[group * 2 + b / 4] >> (8 * (b & 3));
                            out[b] = half2_bits(__hmul2(e2m1x2_to_half2(byte & 0xFFU), scale));
                        }
                        dst[group * 2]     = make_uint4(out[0], out[1], out[2], out[3]);
                        dst[group * 2 + 1] = make_uint4(out[4], out[5], out[6], out[7]);
                    }
                    auto* xdst = reinterpret_cast<uint4*>(x_sh + x_row * kStride + x_col);
                    xdst[0] = make_uint4(bf16x2_to_half2_bits(regs.x[0].x), bf16x2_to_half2_bits(regs.x[0].y),
                                         bf16x2_to_half2_bits(regs.x[0].z), bf16x2_to_half2_bits(regs.x[0].w));
                    xdst[1] = make_uint4(bf16x2_to_half2_bits(regs.x[1].x), bf16x2_to_half2_bits(regs.x[1].y),
                                         bf16x2_to_half2_bits(regs.x[1].z), bf16x2_to_half2_bits(regs.x[1].w));
                }
                __syncthreads();
                if (step + 1 < kSteps) { load(step + 1, regs); }
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    // Gate/up: groups 0,1 are gate pairs [16w, 16w + 16), 2,3 the matching up rows.
                    const int group_row = GateUp ? (g < 2 ? 16 * warp + 8 * g : 64 + 16 * warp + 8 * (g - 2))
                                                 : 32 * warp + 8 * g;
                    ops::detail::turing_tile_mma_step<kKStep, kStride>(d[g], x_sh, w_sh + group_row * kStride);
                }
            }

            // Epilogue.
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int token_row = ops::detail::turing_tile_row(mt, i);
                    if (token_row >= valid) { continue; }
                    const int pos   = start + base + token_row;
                    const int token = groups.grouped_tokens[pos];
                    const int path  = groups.grouped_paths[pos];
                    const int col   = ops::detail::turing_tile_col(i);
                    if constexpr (GateUp) {
#pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            const int pair  = tile * 64 + 16 * warp + 8 * j + col;
                            const float g   = d[j][mt][i] * inverse;
                            const float u   = d[2 + j][mt][i] * inverse;
                            const float act = ops::silu(g) * u;
                            activations[(static_cast<std::int64_t>(token) * kPaths + path) * kIntermediate + pair] =
                                __float2bfloat16_rn(act);
                        }
                    } else {
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int out_row = tile * kRows + 32 * warp + 8 * j + col;
                            down_intermediate[(static_cast<std::int64_t>(token) * kHidden + out_row) * kTopK + path] =
                                d[j][mt][i] * inverse;
                        }
                    }
                }
            }
        }
    }
#endif
}

std::atomic<int> g_override{-1};

} // namespace

bool flash_next_moe_prefill_turing_enabled() {
    const int forced = g_override.load(std::memory_order_relaxed);
    if (forced >= 0) { return forced != 0; }
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_FLASH_NEXT_MOE_TURING");
        if (value != nullptr && std::strcmp(value, "0") == 0) { return false; }
        int device = 0;
        cudaDeviceProp prop{};
        return cudaGetDevice(&device) == cudaSuccess && cudaGetDeviceProperties(&prop, device) == cudaSuccess &&
               prop.major == 7 && prop.minor == 5;
    }();
    return enabled;
}

void flash_next_moe_prefill_turing_override(bool enabled) { g_override.store(enabled ? 1 : 0); }

void flash_next_moe_prefill_turing_gate_up(const FlashNextTuringGroups& groups, const FlashNextTuringBank& bank,
                                           const void* input_bf16, void* activations_bf16, cudaStream_t stream) {
    flash_next_moe_prefill_turing_kernel<true><<<dim3(kIntermediate / 64, kGridY), kThreads, 0, stream>>>(
        groups, bank, static_cast<const __nv_bfloat16*>(input_bf16), static_cast<__nv_bfloat16*>(activations_bf16),
        nullptr);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_moe_prefill_turing_down(const FlashNextTuringGroups& groups, const FlashNextTuringBank& bank,
                                        const void* activations_bf16, float* down_intermediate, cudaStream_t stream) {
    flash_next_moe_prefill_turing_kernel<false><<<dim3(kHidden / kRows, kGridY), kThreads, 0, stream>>>(
        groups, bank, static_cast<const __nv_bfloat16*>(activations_bf16), nullptr, down_intermediate);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
