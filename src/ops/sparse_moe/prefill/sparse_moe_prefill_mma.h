#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Grouped tensor-core routed projections for the sm_70 build's prefill (Volta mma.m8n8k4 on 7.0,
// Turing mma.m16n8k8 on 7.5; compiled for both in their own archive). Every CTA is one 32-row
// block of one 32-column tile of one route job (an expert's packed column range from the scan)
// and runs the dense Linear tile body with pointers moved to that expert's bank rows and packed
// columns. Jobs past a negative (adaptive-route) job count exit.
inline constexpr int kSparseMoePrefillMmaTile = 32;

// Resolves this archive's kernels (its own non-RDC module) so CUDA's lazy loading does not load
// them inside the first prefill; called from ops::load_device_code().
void sparse_moe_prefill_mma_load_device_code();

void sparse_moe_prefill_q4_gate_up_mma_launch(const __nv_bfloat16* gathered, const int* expert_offsets,
                                              const int* route_job_experts, const int* route_job_columns,
                                              const int* route_job_count, const int* bank_of_expert,
                                              int max_route_jobs, int tiles_per_job, const std::uint8_t* codes,
                                              const std::uint8_t* scales, __nv_bfloat16* gate_up,
                                              cudaStream_t stream);

void sparse_moe_prefill_q5_down_mma_launch(const __nv_bfloat16* activation, const int* expert_offsets,
                                           const int* route_job_experts, const int* route_job_columns,
                                           const int* route_job_count, const int* bank_of_expert,
                                           int max_route_jobs, int tiles_per_job, const std::uint8_t* codes,
                                           const std::uint8_t* high, const std::uint8_t* scales,
                                           __nv_bfloat16* output, cudaStream_t stream);

} // namespace ninfer::ops::detail
