#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_small_t.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename Bf16LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);

    const Bf16SmallTContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                            Geometry::kOutputRows};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    bf16_small_t_inner_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.qdata), output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Bf16Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, kBf16SmallTMinTokens + static_cast<int>(Offsets)>...};
}

using ControlGeometry = Bf16GemvGeometry<14336, 5120>;
using OutputGeometry  = Bf16GemvGeometry<5120, 6144>;

constexpr auto kControlLaunchers = make_launchers<ControlGeometry>(
    std::make_index_sequence<kBf16SmallTMaxTokens - kBf16SmallTMinTokens + 1>{});
constexpr auto kOutputLaunchers = make_launchers<OutputGeometry>(
    std::make_index_sequence<kBf16SmallTMaxTokens - kBf16SmallTMinTokens + 1>{});
// Qwen3.8-Flash-Next shapes (K = 2560).
constexpr auto kNgramKeyLaunchers = make_launchers<Bf16GemvGeometry<10240, 2560>>(
    std::make_index_sequence<kBf16SmallTMaxTokens - kBf16SmallTMinTokens + 1>{});
constexpr auto kNgramValueLaunchers = make_launchers<Bf16GemvGeometry<2560, 2560>>(
    std::make_index_sequence<kBf16SmallTMaxTokens - kBf16SmallTMinTokens + 1>{});
constexpr auto kIndexerLaunchers = make_launchers<Bf16GemvGeometry<640, 2560>>(
    std::make_index_sequence<kBf16SmallTMaxTokens - kBf16SmallTMinTokens + 1>{});

} // namespace

void launch_bf16_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kBf16SmallTMinTokens);
    if (weight.n == ControlGeometry::kOutputRows && weight.k == ControlGeometry::kInputRows) {
        kControlLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == OutputGeometry::kOutputRows && weight.k == OutputGeometry::kInputRows) {
        kOutputLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.k == 2560) {
        const auto* table = weight.n == 10240 ? &kNgramKeyLaunchers
                            : weight.n == 2560 ? &kNgramValueLaunchers
                            : weight.n == 640  ? &kIndexerLaunchers
                                               : nullptr;
        if (table != nullptr) {
            (*table)[index](x, weight, out, stream);
            return;
        }
    }
    throw std::invalid_argument("bf16 linear small-T: unsupported exact problem");
}

} // namespace ninfer::ops::detail
