#pragma once

namespace ninfer::ops {

/**
 * Loads the Op library's device code now instead of at its first kernel launch.
 *
 * The Ops are one device-linked module; under CUDA's default lazy loading the first launch of
 * any of its kernels loads that whole image, which costs hundreds of milliseconds and device
 * memory. Programs call this during creation so neither lands inside a request (measured 0.7 s
 * inside a 546-token 35B-A3B prefill) or after the memory plan has been sized.
 */
void load_device_code();

} // namespace ninfer::ops
