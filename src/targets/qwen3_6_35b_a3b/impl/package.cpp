#include <ninfer/targets/qwen3_6_35b_a3b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "artifact/reader.h"
#include "targets/qwen3_6_35b_a3b/impl/load/bindings.h"
#include "targets/qwen3_6_35b_a3b/impl/variant.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6_35b_a3b::detail {

class LoadPlan::Impl {
public:
    Impl(WeightsProfile weights_profile_in, ArtifactLoadPlan target_plan)
        : weights_profile(weights_profile_in), plan(std::move(target_plan)) {}

    WeightsProfile weights_profile;
    ArtifactLoadPlan plan;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->plan.materialization;
}

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadedModel::~LoadedModel() = default;

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail

namespace ninfer::targets::qwen3_6_35b_a3b {
namespace {

// Device memory the expert cache leaves to allocations made after Program creation (CUDA
// context growth, cuBLAS/CUTLASS handles, transient host-pinned staging) when sized automatically.
constexpr std::size_t kOffloadHeadroomBytes = 384ULL << 20;

// Offloaded experts need a host synchronization inside every SparseMoe call, which CUDA Graph
// capture cannot contain.
EngineOptions effective_options(const EngineOptions& options) {
    EngineOptions out = options;
    if (options.offload_routed_experts || options.routed_expert_cache_bytes != 0) {
        out.offload_routed_experts = true;
        out.use_cuda_graph         = false;
    }
    return out;
}

constexpr ModelSamplingDefaults kQwen3_6_35BA3BDefaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
};

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kQwen3_6_35BA3BDefaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == "groupwise-int") {
        return WeightsProfile::GroupwiseInt;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options_in,
                                     WeightsProfile weights_profile) {
    const EngineOptions options = effective_options(options_in);
    detail::ExpertOffloadOptions offload{};
    if (options.offload_routed_experts) {
#ifndef NINFER_VOLTA_BUILD
        throw std::invalid_argument("--offload-experts requires the sm_70 build");
#endif
        offload.enabled     = true;
        offload.cache_bytes = options.routed_expert_cache_bytes;
        // Widest SparseMoe call: a prefill chunk, a verified draft window, or one decode round.
        offload.max_tokens = static_cast<std::int32_t>(std::max<std::uint32_t>(
            {options.prefill_chunk, options.max_concurrency * (options.speculative.draft_tokens + 1), 1U}));
    }
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile, detail::bind_artifact(binder, qwen3_6::startup_features(options), offload)));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }
    auto impl = std::make_unique<LoadedModel::Impl>(
        plan.impl_->weights_profile, std::move(plan.impl_->plan.bindings), std::move(materialized));
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::make_frontend(model.impl_->data.frontend,
                                  qwen3_6::FrontendOptions{
                                      .vision_enabled = model.impl_->data.runtime.features.vision,
                                      .max_context    = options.max_context,
                                      .media_cache_bytes        = options.media_cache_bytes,
                                      .media_live_bytes         = options.media_live_bytes,
                                      .media_preprocess_threads = options.media_preprocess_threads,
                                  });
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile) {
    return qwen3_6::make_sequence_planner<detail::Variant>(device, effective_options(options),
                                                           weights_profile);
}

std::unique_ptr<Package::Program> Package::create_program(const LoadedModel& model,
                                                          SequencePlan&& plan,
                                                          DeviceContext& device,
                                                          const StartupObserver& startup_observer) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    auto program = qwen3_6::create_program<detail::Variant>(model.impl_->data.runtime,
                                                            model.impl_->weights_profile,
                                                            std::move(plan), device, startup_observer);
    if (detail::ExpertOffload* offload = model.impl_->data.offload.get(); offload != nullptr) {
        // The cache takes what the weights and the Program's runtime reservation leave.
        std::size_t budget = offload->requested_bytes();
        if (budget == 0) {
            device.synchronize();
            std::size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            budget = free_bytes > kOffloadHeadroomBytes ? free_bytes - kOffloadHeadroomBytes : 0;
        }
        offload->allocate(budget);
    }
    return program;
}

} // namespace ninfer::targets::qwen3_6_35b_a3b
