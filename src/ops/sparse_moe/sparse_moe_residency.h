#pragma once

#include "ninfer/ops/sparse_moe.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The one host synchronization of an offloaded SparseMoe call, placed between routing and the
// routed expert kernels. Copies the `count` selected ids at `device_ids` to the host, lets the
// residency make them resident, then either rewrites `device_ids` in place with bank indices
// (paths that use the ids only to address expert rows) or uploads the 256-entry expert-to-bank
// table to `device_bank_of_expert` (paths that also group tokens by expert id).
void resolve_sparse_moe_residency(const SparseMoeExpertResidency& residency, int* device_ids,
                                  std::int32_t count, bool remap_ids, int* device_bank_of_expert,
                                  cudaStream_t stream);

// Decode/small-T form with host-computed cold experts: also copies the route weights and the BF16
// input to the host in the same synchronization, maps cold experts to the zero bank, and returns
// whether any selected expert is cold.
[[nodiscard]] bool resolve_sparse_moe_residency_cold(const SparseMoeExpertResidency& residency,
                                                     int* device_ids, const float* device_alpha,
                                                     const void* device_x, std::int32_t tokens,
                                                     cudaStream_t stream);

// Prefill form (T <= residency.prefill_cold_max_tokens): copies ids, route weights and input to
// the host, lets acquire split the missing experts between uploads and the host (which leaves
// their sum in residency.host_cold_sum), and uploads the expert-to-bank table with cold experts
// on the zero bank. Returns whether any selected expert is cold.
[[nodiscard]] bool resolve_sparse_moe_residency_prefill(const SparseMoeExpertResidency& residency,
                                                        const int* device_ids, const float* device_alpha,
                                                        const void* device_x, std::int32_t tokens,
                                                        int* device_bank_of_expert, cudaStream_t stream);

// Uploads residency.host_cold_sum [2048 * tokens] FP32 to device_cold_sum, ordered on stream.
void upload_sparse_moe_cold_sum(const SparseMoeExpertResidency& residency, std::int32_t tokens,
                                float* device_cold_sum, cudaStream_t stream);

// Computes the cold experts of the last resolve on the host and uploads their sum to
// device_cold_sum [2048 * tokens] FP32, ordered on stream.
void finish_sparse_moe_cold(const SparseMoeExpertResidency& residency, std::int32_t tokens,
                            float* device_cold_sum, cudaStream_t stream);

} // namespace ninfer::ops::detail
