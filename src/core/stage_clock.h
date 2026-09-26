#pragma once

// NINFER_STAGE_CLOCK: diagnostic device-time laps for prefill. Every lap synchronizes the
// stream, so the totals (printed at exit) are only meaningful as a breakdown.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

namespace ninfer::stage_clock {

struct State {
    bool enabled = std::getenv("NINFER_STAGE_CLOCK") != nullptr;
    cudaEvent_t last = nullptr;
    std::map<std::string, double> totals;
    ~State() {
        double sum = 0;
        for (const auto& [name, s] : totals) { sum += s; }
        for (const auto& [name, s] : totals) {
            std::fprintf(stderr, "[stage clock] %-28s %8.3f s (%4.1f%%)\n", name.c_str(), s,
                         sum > 0 ? 100.0 * s / sum : 0.0);
        }
    }
};

inline State& state() {
    static State s;
    return s;
}

inline bool enabled() { return state().enabled; }

// Restarts the clock without charging anything.
inline void reset(cudaStream_t stream) {
    State& s = state();
    if (!s.enabled) { return; }
    if (s.last == nullptr) { cudaEventCreate(&s.last); }
    cudaEventRecord(s.last, stream);
}

// Charges the device time since the previous lap/reset to `name`.
inline void lap(cudaStream_t stream, const char* name) {
    State& s = state();
    if (!s.enabled || s.last == nullptr) { return; }
    cudaEvent_t now = nullptr;
    cudaEventCreate(&now);
    cudaEventRecord(now, stream);
    cudaEventSynchronize(now);
    float ms = 0;
    cudaEventElapsedTime(&ms, s.last, now);
    s.totals[name] += ms / 1e3;
    cudaEventDestroy(s.last);
    s.last = now;
}

} // namespace ninfer::stage_clock
