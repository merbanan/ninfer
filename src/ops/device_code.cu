#include "ninfer/ops/device_code.h"

#include "core/device.h"

#include <cuda_runtime.h>

namespace ninfer::ops {
namespace {

// Any kernel of this device-linked module resolves the whole module image.
__global__ void device_code_anchor_kernel() {}

} // namespace

void load_device_code() {
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, device_code_anchor_kernel));
}

} // namespace ninfer::ops
