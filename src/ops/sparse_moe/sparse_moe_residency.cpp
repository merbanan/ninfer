#include "ops/sparse_moe/sparse_moe_residency.h"

#include "core/device.h"

#include <span>
#include <stdexcept>

namespace ninfer::ops::detail {

void resolve_sparse_moe_residency(const SparseMoeExpertResidency& residency, int* device_ids,
                                  std::int32_t count, bool remap_ids, int* device_bank_of_expert,
                                  cudaStream_t stream) {
    constexpr std::int32_t kExperts = 256;
    std::int32_t* host_ids          = residency.host_ids;
    std::int32_t* bank_of_expert    = residency.host_bank_of_expert;
    CUDA_CHECK(cudaMemcpyAsync(host_ids, device_ids, sizeof(std::int32_t) * count,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::int32_t i = 0; i < kExperts; ++i) { bank_of_expert[i] = -1; }
    residency.acquire(std::span<const std::int32_t>(host_ids, static_cast<std::size_t>(count)),
                      bank_of_expert, stream);
    for (std::int32_t i = 0; i < count; ++i) {
        const std::int32_t expert = host_ids[i];
        if (expert < 0 || expert >= kExperts || bank_of_expert[expert] < 0 ||
            bank_of_expert[expert] >= residency.banks) {
            throw std::logic_error("sparse_moe: expert residency left a selected expert unmapped");
        }
    }
    if (remap_ids) {
        // Stream-ordered: the next call's device-to-host copy into host_ids, and its host read
        // after the synchronization, both follow this upload.
        for (std::int32_t i = 0; i < count; ++i) { host_ids[i] = bank_of_expert[host_ids[i]]; }
        CUDA_CHECK(cudaMemcpyAsync(device_ids, host_ids, sizeof(std::int32_t) * count,
                                   cudaMemcpyHostToDevice, stream));
    }
    if (device_bank_of_expert != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(device_bank_of_expert, bank_of_expert,
                                   sizeof(std::int32_t) * kExperts, cudaMemcpyHostToDevice, stream));
    }
}

} // namespace ninfer::ops::detail
