#pragma once

#include "sort-common.cuh"

// #define __CUDA_ARCH__ 700
// #define __CUDA_ARCH__ 610

gpu u32 match_any_polyfill(u32 thread_mask, u32 key) {
#if __CUDA_ARCH__ >= 700
  // sm_70+ volta, turing, ...
  return __match_any_sync(thread_mask, key);
#else
  u32 candidates = thread_mask;
  for (u32 mask = 1; mask < RADIX; mask <<= 1) {
    u32 bit = key & mask;
    candidates &= __ballot_sync(thread_mask, static_cast<int>(bit)) ^ (UINT_MAX * !bit);
  }
  return candidates;
#endif
}

template <typename T, typename Hasher>
  requires is_hasher<Hasher, T>
global void onesweep(u32 *const atomic_id,
                     T *const data,
                     u32 const length,
                     u32 const place,
                     u32 *const status_counters,
                     u32 (*const totals)[RADIX],
                     Hasher hasher) {
  constexpr u32 subwarps_per_warp = 32 / sorting::subwarp_size;
  constexpr u32 subwarps = sorting::warp_slices * subwarps_per_warp;

  // RADIX is multiple of 32, thus row reads are guaranteed to conflict
  // RADIX + subwarps_per_warp offsets the rows by subwarp slice
  shared u32 counters[sorting::warp_slices][RADIX + subwarps_per_warp]; // NOLINT(bugprone-dynamic-static-initializers)
  auto &linearized_counters = *reinterpret_cast<u32(*)[ARRAY_LENGTH_2D(counters)]>(counters);

  for (u32 i = threadIdx.x; i < ARRAY_LENGTH(linearized_counters); i += blockDim.x) {
    linearized_counters[i] = 0;
  }

  shared u32 block_id; // NOLINT(bugprone-dynamic-static-initializers)
  if (threadIdx.x == 0) {
    block_id = atomicAdd(atomic_id, 1);
  }
  __syncthreads();

  u32 const lane = threadIdx.x % 32;
  u32 const warp_id = threadIdx.x / 32;

  constexpr u32 warp_slice_length = 32 * sorting::items_per_thread;
  u32 const warp_slice_start = block_id * sorting::slice_length + warp_id * warp_slice_length;
  u32 const warp_slice_end = min(length, warp_slice_start + warp_slice_length);

  // iterate over the data slice, partitioned by warps
  for (u32 i = warp_slice_start; i + lane < warp_slice_end; i += 32) {
    T const item = data[i + lane];
    u32 const hash = hasher(item);
    u32 const digit = (hash >> place * RADIX_SIZE) & RADIX_MASK;

    u32 const warp_usage = min(warp_slice_end - i, 32);
    u32 const thread_mask = __funnelshift_lc(UINT_MAX, 0, warp_usage);

    u32 const matches = match_any_polyfill(thread_mask, digit);
    u32 const leader_id = __ffs(matches);

    u32 exclusive_sum = 0;

    if (lane == leader_id) {
      exclusive_sum = counters[warp_id][digit];
      counters[warp_id][digit] += __popc(matches);
    }

    exclusive_sum = __shfl_sync(matches, exclusive_sum, leader_id);
    exclusive_sum += __popc(matches & (1 << lane) - 1);
  }

  u32 const subwarp_id = threadIdx.x / sorting::subwarp_size;
  u32 const subwarp_lane = threadIdx.x % sorting::subwarp_size;

  for (u32 i = subwarp_id; i < ARRAY_LENGTH(*counters); i += subwarps) {
    u32 val = subwarp_lane < sorting::warp_slices ? counters[subwarp_lane][i] : 0;
    val = warp_inclusive_multiscan<sorting::subwarp_size>(val);
    if (subwarp_lane < sorting::warp_slices) {
      counters[subwarp_lane][i] = val;
    }
  }
}
