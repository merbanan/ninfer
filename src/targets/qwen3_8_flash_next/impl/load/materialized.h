#pragma once

#include "artifact/materializer.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// Pinned host memory mapped into the device address space (read by kernels over PCIe).
struct MappedHostBuffer {
    void* host         = nullptr;
    void* device       = nullptr;
    std::size_t bytes  = 0;
    MappedHostBuffer() = default;
    explicit MappedHostBuffer(std::size_t size);
    ~MappedHostBuffer();
    MappedHostBuffer(const MappedHostBuffer&)            = delete;
    MappedHostBuffer& operator=(const MappedHostBuffer&) = delete;
};

class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized,
                    bool quantize_output_head_fp8 = false,
                    bool quantize_token_embedding_fp8 = false);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    DeviceBuffer output_head_fp8;
    DeviceBuffer token_embedding_fp8;
    // The FP8 token embedding in host memory instead (sm_70 build on small cards, see
    // flash_next_host_embedding): the gather reads one 2.5 KB row per token over PCIe.
    std::unique_ptr<MappedHostBuffer> token_embedding_host;
    DeviceBuffer mtp_expert_gate_up_nvfp4;
    DeviceBuffer mtp_expert_down_nvfp4;
    DeviceBuffer proposal_head_payload;
    DeviceBuffer proposal_head_fp8;
    DeviceBuffer proposal_token_ids;
    qwen3_6::FrontendResources frontend;
    TextModelView text;
    std::optional<VisionModelView> vision;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
