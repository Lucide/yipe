#pragma once

#include <cuda/std/optional>
#include "gpu-common.cuh"

// intra-warp inclusive prefix sum using register shuffles
template <u32 width> gpu inline u32 warp_inclusive_multiscan(u32 val) {
  static_assert(cuda::std::has_single_bit(width), "Width of warp_inclusive_scan must be a power of two");
  static_assert(width <= 32, "Width of warp_inclusive_scan must <= 32");
#pragma unroll
  for (u32 offset = 1; offset < width; offset <<= 1) {
    u32 const remote_val = __shfl_up_sync(SYNC_WARP, val, offset, width);
    if ((threadIdx.x % width) >= offset) {
      val += remote_val;
    }
  }
  return val;
}
