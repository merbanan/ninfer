#include "ops/linear/w8/w8_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/w8/w8_rowsplit_storage.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

// Eight codes per thread; `padded_k` is the stored row length (codes and scale groups).
__global__ void w8_dequant_to_fp16_kernel(const std::uint8_t* __restrict__ codes,
                                          const std::uint8_t* __restrict__ scales, int rows, int k,
                                          int padded_k, cutlass::half_t* __restrict__ out) {
    const std::int64_t i = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    if (i >= static_cast<std::int64_t>(rows) * k) { return; }
    const std::int64_t row = i / k;
    const int col          = static_cast<int>(i - row * k);
    const std::int64_t group =
        row * (padded_k / W8RowSplitStorage::kGroupK) + col / W8RowSplitStorage::kGroupK;
    const __half scale = __ushort_as_half(reinterpret_cast<const std::uint16_t*>(scales)[group]);
    const uint2 packed = *reinterpret_cast<const uint2*>(codes + row * padded_k + col);
    const auto* c      = reinterpret_cast<const std::int8_t*>(&packed);
    __half h[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) { h[e] = __hmul(__int2half_rn(c[e]), scale); }
    *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<const uint4*>(h);
}

__global__ void w8_bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ in,
                                       cutlass::half_t* __restrict__ out, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = cutlass::half_t(__bfloat162float(in[i])); }
}

using ElementAccumulator = float;
using ElementOutput      = cutlass::bfloat16_t;
using EpilogueOp =
    cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                 128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                 ElementAccumulator, ElementAccumulator>;
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::RowMajor, ElementAccumulator, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70, cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<8, 8, 4>, EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

typename Gemm::Arguments arguments_for(const cutlass::half_t* x, const cutlass::half_t* w, void* out,
                                       std::int32_t out_ld, std::int32_t n, std::int32_t k,
                                       std::int32_t cols, bool accumulate) {
    auto* o = static_cast<ElementOutput*>(out);
    return typename Gemm::Arguments{{cols, n, k},
                                    {x, k},
                                    {w, k},
                                    {o, out_ld},
                                    {o, out_ld},
                                    {ElementAccumulator(1), ElementAccumulator(accumulate ? 1 : 0)},
                                    1};
}

template <class Allocator>
struct Scratch {
    Tensor w_fp16;
    Tensor x_fp16;
    DeviceSpan gemm_workspace;
};

template <class Allocator>
Scratch<Allocator> allocate(Allocator& allocator, std::int32_t n, std::int32_t k, std::int32_t cols,
                            std::size_t gemm_bytes) {
    Scratch<Allocator> out;
    out.w_fp16 = allocator.alloc(DType::FP16, {k, n});
    out.x_fp16 = allocator.alloc(DType::FP16, {k, cols});
    if (gemm_bytes > 0) { out.gemm_workspace = allocator.alloc_bytes(gemm_bytes); }
    return out;
}

std::size_t gemm_bytes(std::int32_t n, std::int32_t k, std::int32_t cols) {
    return Gemm::get_workspace_size(arguments_for(nullptr, nullptr, nullptr, n, n, k, cols, false));
}

} // namespace

std::size_t w8_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate(layout, n, k, cols, gemm_bytes(n, k, cols));
    return layout.peak_bytes(1);
}

bool w8_cutlass_sm70_fits(const WorkspaceArena& ws, std::int32_t n, std::int32_t k,
                          std::int32_t cols) {
    const std::size_t start = (ws.used() + 255) / 256 * 256;
    return ws.capacity() >= start + w8_cutlass_sm70_workspace_bytes(n, k, cols);
}

void w8_cutlass_sm70_run(const Tensor& x, const Weight& weight,
                         std::span<const W8CutlassSegment> segments, bool accumulate,
                         WorkspaceArena& ws, cudaStream_t stream) {
    const std::int32_t n        = weight.n;
    const std::int32_t k        = weight.k;
    const std::int32_t padded_k = weight.padded_shape[1];
    const std::int32_t cols     = x.ne[1];
    if (x.ne[0] != k || k % 8 != 0 || padded_k % W8RowSplitStorage::kGroupK != 0) {
        throw std::invalid_argument("w8_cutlass_sm70: unsupported shape");
    }
    auto scope = ws.scope();
    std::size_t gbytes = 0;
    for (const auto& segment : segments) {
        const std::size_t b = gemm_bytes(segment.rows, k, cols);
        gbytes              = b > gbytes ? b : gbytes;
    }
    Scratch<WorkspaceArena> scratch = allocate(ws, n, k, cols, gbytes);
    auto* w_fp16 = static_cast<cutlass::half_t*>(scratch.w_fp16.data);
    auto* x_fp16 = static_cast<cutlass::half_t*>(scratch.x_fp16.data);

    const std::int64_t w_vectors = static_cast<std::int64_t>(n) * k / 8;
    w8_dequant_to_fp16_kernel<<<static_cast<unsigned>((w_vectors + 255) / 256), 256, 0, stream>>>(
        static_cast<const std::uint8_t*>(weight.qdata), static_cast<const std::uint8_t*>(weight.scales),
        n, k, padded_k, w_fp16);
    CUDA_CHECK(cudaGetLastError());
    const std::int64_t count = static_cast<std::int64_t>(cols) * k;
    w8_bf16_to_fp16_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), x_fp16, count);
    CUDA_CHECK(cudaGetLastError());

    for (const auto& segment : segments) {
        Gemm gemm;
        const auto arguments =
            arguments_for(x_fp16, w_fp16 + static_cast<std::int64_t>(segment.row0) * k, segment.out,
                          segment.out_ld, segment.rows, k, cols, accumulate);
        cutlass::Status status = gemm.can_implement(arguments);
        if (status == cutlass::Status::kSuccess) {
            status = gemm.initialize(arguments, scratch.gemm_workspace.data, stream);
        }
        if (status == cutlass::Status::kSuccess) { status = gemm(stream); }
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error(std::string("w8_cutlass_sm70: CUTLASS failed: ") +
                                     cutlassGetStatusString(status));
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
