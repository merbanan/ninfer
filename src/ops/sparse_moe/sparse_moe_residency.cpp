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
                      bank_of_expert, false, stream);
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

bool resolve_sparse_moe_residency_cold(const SparseMoeExpertResidency& residency, int* device_ids,
                                       const float* device_alpha, const void* device_x, std::int32_t tokens,
                                       cudaStream_t stream) {
    constexpr std::int32_t kExperts = 256;
    constexpr std::int32_t kTopK    = 8;
    constexpr std::int32_t kHidden  = 2048;
    const std::int32_t count        = tokens * kTopK;
    std::int32_t* host_ids          = residency.host_ids;
    std::int32_t* bank_of_expert    = residency.host_bank_of_expert;
    CUDA_CHECK(cudaMemcpyAsync(host_ids, device_ids, sizeof(std::int32_t) * count, cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(residency.host_alpha, device_alpha, sizeof(float) * count,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(residency.host_x, device_x, sizeof(std::uint16_t) * kHidden * tokens,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::int32_t i = 0; i < kExperts; ++i) { bank_of_expert[i] = -1; }
    residency.acquire(std::span<const std::int32_t>(host_ids, static_cast<std::size_t>(count)), bank_of_expert,
                      true, stream);
    bool cold = false;
    for (std::int32_t i = 0; i < count; ++i) {
        const std::int32_t expert = host_ids[i];
        const std::int32_t bank   = expert >= 0 && expert < kExperts ? bank_of_expert[expert] : -2;
        if (bank == -1) {
            cold = true;
        } else if (bank < 0 || bank >= residency.banks) {
            throw std::logic_error("sparse_moe: expert residency left a selected expert unmapped");
        }
    }
    if (cold && (residency.zero_bank < 0 || residency.zero_bank >= residency.banks)) {
        throw std::logic_error("sparse_moe: cold experts need a zero bank");
    }
    // The host ids stay expert ids for cold_compute; the device copy gets bank indices.
    std::int32_t* device_banks = residency.host_ids + count; // scratch tail of host_ids
    for (std::int32_t i = 0; i < count; ++i) {
        const std::int32_t bank = bank_of_expert[host_ids[i]];
        device_banks[i]         = bank < 0 ? residency.zero_bank : bank;
    }
    CUDA_CHECK(cudaMemcpyAsync(device_ids, device_banks, sizeof(std::int32_t) * count, cudaMemcpyHostToDevice,
                               stream));
    return cold;
}

void finish_sparse_moe_cold(const SparseMoeExpertResidency& residency, std::int32_t tokens, float* device_cold_sum,
                            cudaStream_t stream) {
    constexpr std::int32_t kTopK   = 8;
    constexpr std::int32_t kHidden = 2048;
    residency.cold_compute(std::span<const std::int32_t>(residency.host_ids, static_cast<std::size_t>(tokens) * kTopK),
                           residency.host_alpha, residency.host_x, tokens, residency.host_bank_of_expert,
                           residency.host_cold_sum);
    CUDA_CHECK(cudaMemcpyAsync(device_cold_sum, residency.host_cold_sum, sizeof(float) * kHidden * tokens,
                               cudaMemcpyHostToDevice, stream));
}

} // namespace ninfer::ops::detail
