#pragma once

#include "gpu-common.cuh"

// intra-warp inclusive prefix sum using register shuffles
template <u8 width = 32, std::integral Data>
GPU inline Data warp_inclusive_multiscan(Data val, u32 const thread_mask = SYNC_WARP) {
  static_assert(cuda::std::has_single_bit(width), "Width of warp_inclusive_scan must be a power of two");
  static_assert(width <= 32, "Width of warp_inclusive_scan must <= 32");
#pragma unroll
  for (u32 offset = 1; offset < width; offset <<= 1) {
    Data const remote_val = __shfl_up_sync(thread_mask, val, offset, width);
    if ((threadIdx.x % width) >= offset) {
      val += remote_val;
    }
  }
  return val;
}
