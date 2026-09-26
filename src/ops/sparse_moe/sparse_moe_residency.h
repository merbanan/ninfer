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

} // namespace ninfer::ops::detail
