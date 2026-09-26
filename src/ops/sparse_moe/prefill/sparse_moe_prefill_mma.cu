#include "ops/sparse_moe/prefill/sparse_moe_prefill_mma.h"

#include "core/device.h"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q4/q4_volta_mma_gemm.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_volta_mma_gemm.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kMmaTile      = kSparseMoePrefillMmaTile;
constexpr int kHidden       = 2048;
constexpr int kIntermediate = 512;

__device__ __forceinline__ bool moe_mma_job(const int* __restrict__ expert_offsets,
                                            const int* __restrict__ route_job_experts,
                                            const int* __restrict__ route_job_columns,
                                            const int* __restrict__ route_job_count,
                                            const int* __restrict__ bank_of_expert, int tiles_per_job,
                                            int& bank, int& begin, int& count, int& tile) {
    const int job = static_cast<int>(blockIdx.z) / tiles_per_job;
    if (job >= *route_job_count) { return false; }
    const int expert = route_job_experts[job];
    begin            = expert_offsets[expert];
    count            = expert_offsets[expert + 1] - begin;
    const int column = route_job_columns[job] + (static_cast<int>(blockIdx.z) % tiles_per_job) * kMmaTile;
    if (column >= count) { return false; }
    tile = column / kMmaTile;
    bank = bank_of_expert != nullptr ? bank_of_expert[expert] : expert;
    return true;
}

static __global__ __launch_bounds__(Q4VoltaMmaSchedule::kThreads, 8) void sparse_moe_prefill_q4_gate_up_mma_kernel(
    const __nv_bfloat16* __restrict__ gathered, const int* __restrict__ expert_offsets,
    const int* __restrict__ route_job_experts, const int* __restrict__ route_job_columns,
    const int* __restrict__ route_job_count, const int* __restrict__ bank_of_expert, int tiles_per_job,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ gate_up) {
    int bank = 0, begin = 0, count = 0, tile = 0;
    if (!moe_mma_job(expert_offsets, route_job_experts, route_job_columns, route_job_count, bank_of_expert,
                     tiles_per_job, bank, begin, count, tile)) {
        return;
    }
    constexpr int kRows   = 2 * kIntermediate;
    constexpr int kGroups = kHidden / Q4RowSplitStorage::kGroupK;
    const std::int64_t row0 = static_cast<std::int64_t>(bank) * kRows * kGroups;
    q4_volta_mma_gemm_tile<true>(codes + row0 * Q4RowSplitStorage::kCodeBytesPerGroup,
                                 scales + row0 * Q4RowSplitStorage::kScaleBytesPerGroup,
                                 gathered + static_cast<std::int64_t>(begin) * kHidden, nullptr,
                                 gate_up + static_cast<std::int64_t>(begin) * kRows, kRows, kRows, kHidden,
                                 count, kGroups, 1, static_cast<int>(blockIdx.x), 0, tile);
}

static __global__ __launch_bounds__(Q5VoltaMmaSchedule::kThreads, 8) void sparse_moe_prefill_q5_down_mma_kernel(
    const __nv_bfloat16* __restrict__ activation, const int* __restrict__ expert_offsets,
    const int* __restrict__ route_job_experts, const int* __restrict__ route_job_columns,
    const int* __restrict__ route_job_count, const int* __restrict__ bank_of_expert, int tiles_per_job,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ high,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ output) {
    int bank = 0, begin = 0, count = 0, tile = 0;
    if (!moe_mma_job(expert_offsets, route_job_experts, route_job_columns, route_job_count, bank_of_expert,
                     tiles_per_job, bank, begin, count, tile)) {
        return;
    }
    constexpr int kGroups   = kIntermediate / Q5RowSplitStorage::kGroupK;
    const std::int64_t row0 = static_cast<std::int64_t>(bank) * kHidden * kGroups;
    q5_volta_mma_gemm_tile<true, false>(codes + row0 * Q5RowSplitStorage::kCodeBytesPerGroup,
                                        high + row0 * Q5RowSplitStorage::kHighBytesPerGroup,
                                        scales + row0 * Q5RowSplitStorage::kScaleBytesPerGroup,
                                        activation + static_cast<std::int64_t>(begin) * kIntermediate, nullptr,
                                        output + static_cast<std::int64_t>(begin) * kHidden, kHidden, kHidden,
                                        kIntermediate, count, kGroups, 1, static_cast<int>(blockIdx.x), 0, tile);
}


} // namespace

void sparse_moe_prefill_mma_load_device_code() {
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, sparse_moe_prefill_q4_gate_up_mma_kernel));
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, sparse_moe_prefill_q5_down_mma_kernel));
}

void sparse_moe_prefill_q4_gate_up_mma_launch(const __nv_bfloat16* gathered, const int* expert_offsets,
                                              const int* route_job_experts, const int* route_job_columns,
                                              const int* route_job_count, const int* bank_of_expert,
                                              int max_route_jobs, int tiles_per_job, const std::uint8_t* codes,
                                              const std::uint8_t* scales, __nv_bfloat16* gate_up,
                                              cudaStream_t stream) {
    sparse_moe_prefill_q4_gate_up_mma_kernel<<<dim3(2 * kIntermediate / Q4VoltaMmaSchedule::kRowsPerCta, 1,
                                                    max_route_jobs * tiles_per_job),
                                               Q4VoltaMmaSchedule::kThreads, 0, stream>>>(
        gathered, expert_offsets, route_job_experts, route_job_columns, route_job_count, bank_of_expert,
        tiles_per_job, codes, scales, gate_up);
    CUDA_CHECK(cudaGetLastError());
}

void sparse_moe_prefill_q5_down_mma_launch(const __nv_bfloat16* activation, const int* expert_offsets,
                                           const int* route_job_experts, const int* route_job_columns,
                                           const int* route_job_count, const int* bank_of_expert,
                                           int max_route_jobs, int tiles_per_job, const std::uint8_t* codes,
                                           const std::uint8_t* high, const std::uint8_t* scales,
                                           __nv_bfloat16* output, cudaStream_t stream) {
    sparse_moe_prefill_q5_down_mma_kernel<<<dim3(kHidden / Q5VoltaMmaSchedule::kRowsPerCta, 1,
                                                 max_route_jobs * tiles_per_job),
                                            Q5VoltaMmaSchedule::kThreads, 0, stream>>>(
        activation, expert_offsets, route_job_experts, route_job_columns, route_job_count, bank_of_expert,
        tiles_per_job, codes, high, scales, output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
