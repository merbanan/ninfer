#pragma once

#include "core/tensor.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace ninfer::ops::cpu {

// Host views of one expert of the 35B-A3B routed banks (row-split-k128-v1 planes, rows of the
// expert only): gate/up [1024 rows: gate 0..511, up 512..1023] x 2048 Q4G64_F16S and down
// [2048 rows] x 512 Q5G64_F16S or Q6G64_F16S.
struct HostExpertView {
    const std::uint8_t* gate_codes  = nullptr; // 1024 rows x 1024 bytes
    const std::uint8_t* gate_scales = nullptr; // 1024 rows x 32 fp16
    const std::uint8_t* down_codes  = nullptr; // 2048 rows x 256 bytes
    const std::uint8_t* down_high   = nullptr; // 2048 rows x 64 (Q5) or 128 (Q6) bytes
    const std::uint8_t* down_scales = nullptr; // 2048 rows x 8 fp16
    QType down_qtype                = QType::Q5G64_F16S;
};

// One routed expert applied to `tokens` input columns, weighted per column.
struct ColdExpertWork {
    HostExpertView expert;
    std::vector<std::int32_t> columns; // token columns routed to this expert
    std::vector<float> weights;        // route weight of each column
};

// Persistent worker pool for the host half of offloaded sparse MoE. Parallel regions are short
// (tens to hundreds of microseconds), so workers spin briefly on a generation counter before
// sleeping.
class SpinPool {
public:
    explicit SpinPool(unsigned threads);
    ~SpinPool();
    SpinPool(const SpinPool&)            = delete;
    SpinPool& operator=(const SpinPool&) = delete;

    [[nodiscard]] unsigned threads() const noexcept { return static_cast<unsigned>(workers_.size()) + 1; }
    // Runs task(i) for i in [0, count) on all threads (the caller participates); returns when done.
    void run(std::size_t count, const std::function<void(std::size_t)>& task);

private:
    void worker_loop();

    std::vector<std::thread> workers_;
    std::atomic<std::uint64_t> generation_{0};
    std::atomic<std::size_t> next_{0};
    std::atomic<std::size_t> done_{0};
    std::atomic<std::size_t> active_{0};
    std::atomic<bool> stop_{false};
    std::size_t count_ = 0;
    const std::function<void(std::size_t)>* task_ = nullptr;
    std::mutex mutex_;
    std::condition_variable wake_;
};

// Sums, into out[column * 2048 + row] (which it first zeroes for every column in [0, tokens)),
// weight * down(silu(gate(x)) * up(x)) of every work item, where x is the BF16 input column
// x_bf16[column * 2048 ...]. Arithmetic is FP32 with exact stored-weight decode, matching the
// device kernels' FP32 path up to summation order.
void cold_experts(std::span<const ColdExpertWork> work, const std::uint16_t* x_bf16, std::int32_t tokens,
                  float* out, SpinPool& pool);

} // namespace ninfer::ops::cpu
