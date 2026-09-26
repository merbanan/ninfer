#include "ninfer/ops/device_code.h"

#include "core/device.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/sparse_moe/prefill/sparse_moe_prefill_mma.h"
#endif

#include <cuda_runtime.h>

namespace ninfer::ops {
namespace {

// Any kernel of this device-linked module resolves the whole module image.
__global__ void device_code_anchor_kernel() {}

} // namespace

void load_device_code() {
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, device_code_anchor_kernel));
#ifdef NINFER_VOLTA_BUILD
    // Separately compiled (sm_70 + sm_75) modules the Op library links.
    detail::sparse_moe_prefill_mma_load_device_code();
#endif
}

} // namespace ninfer::ops
