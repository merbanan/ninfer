#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>

namespace ninfer::ops {

struct SparseMoeWeights {
    Weight router_shared_gate;
    Weight routed_gate_up;
    Weight routed_down;
    Weight shared_gate_up;
    Weight shared_down;
};

enum class SparseMoeEpilogue : std::uint8_t {
    AddResidual,
};

/**
 * Optional per-call execution hints. Every field is a pure cache hint with no numeric effect:
 * the same call with a default-constructed SparseMoeHints produces bit-identical output.
 *
 * next_weight_prefetch names a weight span the next decode-step consumer will stream; the decode
 * D4 epilogue issues fire-and-forget L2 prefetches over its first bytes. The span is caller-owned
 * and read once, inside the call: the Op keeps no state between calls, and no hidden channel
 * carries it.
 */
struct SparseMoeHints {
    const void* next_weight_prefetch       = nullptr;
    std::size_t next_weight_prefetch_bytes = 0;
};

/**
 * Routed experts that live outside device memory (expert offload, sm_70 build only).
 *
 * The routed-bank views in SparseMoeWeights then hold `banks` experts in the registered row
 * geometry ([banks*1024,2048] gate/up, [banks*2048,512] down) instead of all 256. After routing,
 * the Op copies the selected expert ids to `host_ids` and synchronizes `stream` once; `acquire`
 * must make every selected expert resident in some bank, write each selected expert's bank index
 * to `bank_of_expert[expert]` (entries of unselected experts are ignored), and order any copies
 * it needs on `stream`. Routing, tie-breaking, token grouping, and the result are exactly those
 * of the resident Op; only the bank each expert's rows are read from changes.
 */
struct SparseMoeExpertResidency {
    std::int32_t* host_ids            = nullptr; // pinned host storage, >= 8 * T entries
    std::int32_t* host_bank_of_expert = nullptr; // pinned host storage, 256 entries
    std::int32_t banks                = 0;
    // cold_allowed is true only for calls that can take host-computed ("cold") experts; acquire
    // may then leave a selected expert cold by writing -1 for it.
    std::function<void(std::span<const std::int32_t> selected_ids, std::int32_t* bank_of_expert,
                       bool cold_allowed, cudaStream_t stream)>
        acquire;

    // Host-computed cold experts (decode and small-T calls; unused when cold_compute is empty).
    // zero_bank is a bank whose weights are all zero: cold experts are read from it on the device,
    // contributing nothing there. cold_compute receives the selected ids, their route weights, the
    // BF16 input columns, and the bank table, and writes into cold_sum[column * 2048 + row] the sum
    // of weight * expert(x) over the cold experts only; the Op adds it before the final rounding.
    std::int32_t zero_bank   = -1;
    float* host_alpha        = nullptr; // pinned, >= 8 * T entries
    std::uint16_t* host_x    = nullptr; // pinned, >= 2048 * T entries
    float* host_cold_sum     = nullptr; // pinned, >= 2048 * T entries
    std::function<void(std::span<const std::int32_t> selected_ids, const float* alpha,
                       const std::uint16_t* x_bf16, std::int32_t tokens,
                       const std::int32_t* bank_of_expert, float* cold_sum)>
        cold_compute;
};

/**
 * Returns the transient capacity required by SparseMoe for every T in the inclusive
 * [min_tokens,max_tokens] interval. The routed QTypes are the fixed implementation profile.
 * Invalid profiles or intervals throw.
 */
[[nodiscard]] std::size_t sparse_moe_workspace_capacity_bytes(QType routed_gate_up,
                                                              QType routed_down,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

/**
 * Closed sparse-MoE Op for the exact future 35B-A3B geometry.
 *
 * For contiguous BF16 x [2048,T] and destination [2048,T] with T>0, 256 routed experts, top-8
 * selection, and one always-on shared expert, this Op owns router projection and selection,
 * selected routed and shared SwiGLU projections, down projections, their merge, and the
 * AddResidual epilogue independently for every token column. At an exact top-8 boundary tie the
 * lower expert id wins. destination is the only observable mutation: its incoming value is the
 * residual and its outgoing value is the BF16 sparse-MoE result plus that residual.
 *
 * The complete mathematical oracle starts from represented BF16 inputs, exact stored-weight
 * decode, and evaluates the logical formula naively in FP32/FP64. Scores, route weights, expert
 * activations, workspace representation, reduction association, and scale placement are private
 * execution choices rather than semantic rounding boundaries.
 *
 * The five weights have the exact registered shapes: BF16 router/shared gate [257,2048], routed
 * gate/up [256*1024,2048], routed down [256*2048,512], shared gate/up [1024,2048], and shared down
 * [2048,512]. Admitted codec profiles are Q4+Q5, Q4+Q6, and W8+W8 for the two routed banks; both
 * shared banks are W8. Expert e directly selects its stored row spans; no selected-weight gather
 * or repack occurs.
 *
 * Every positive T is supported.
 *
 * x, destination, all weight planes, and live workspace must be pairwise non-overlapping.
 * Execution is enqueued on stream without host synchronization. Workspace is caller-owned,
 * graph-stable transient storage and carries no state beyond the call.
 */
void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * The same Op over offloaded routed experts (see SparseMoeExpertResidency). The Q4+Q5 and Q4+Q6
 * routed profiles are admitted; the workspace requirement is the resident Op's.
 */
void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, const SparseMoeExpertResidency& residency,
                WorkspaceArena& workspace, cudaStream_t stream);

/**
 * The same Op with caller-supplied execution hints. Semantics, workspace requirement and output
 * are exactly those of the overload above; hints only steer cache warming.
 */
void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, const SparseMoeHints& hints, WorkspaceArena& workspace,
                cudaStream_t stream);

} // namespace ninfer::ops
