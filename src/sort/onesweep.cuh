#pragma once

#include <cuda/atomic>
#include <cuda/std/mdspan>
#include "sort-common.cuh"

// #define __CUDA_ARCH__ 700
// #define __CUDA_ARCH__ 610
namespace sorting {

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
kernel void onesweep(u32 *const atomic_id,
                     T *const data,
                     u32 const length,
                     u8 const place,
                     StatusCounter *const linearized_global_status_counters,
                     u32 (*const global_totals)[RADIX],
                     Hasher hasher) {
  constexpr u8 subwarps_per_warp = 32 / subwarp_size;
  constexpr u16 subwarps = warp_slices * subwarps_per_warp;

  // RADIX is multiple of 32, thus row reads are guaranteed to conflict
  // RADIX + subwarps_per_warp offsets the rows by subwarp slice
  shared u16 counters[warp_slices][RADIX + subwarps_per_warp]; // NOLINT(bugprone-dynamic-static-initializers)
  auto &linearized_counters = *reinterpret_cast<u32(*)[ARRAY_LENGTH_2D(counters)]>(counters);

  for (u16 i = threadIdx.x; i < ARRAY_LENGTH(linearized_counters); i += blockDim.x) {
    linearized_counters[i] = 0;
  }

  shared u32 block_id; // NOLINT(bugprone-dynamic-static-initializers)
  if (threadIdx.x == 0) {
    block_id = atomicAdd(atomic_id, 1) % gridDim.x;
  }
  __syncthreads();

  u8 const lane = threadIdx.x % 32;
  u16 const warp_id = threadIdx.x / 32;

  u32 const warp_slice_start = block_id * slice_length + warp_id * warp_slice_length;
  u32 const warp_slice_end = min(length, warp_slice_start + warp_slice_length);

  u16 exclusive_sums[items_per_thread]; // NOLINT(bugprone-dynamic-static-initializers)

  // iterate over the data slice, partitioned by warps
  for (u32 i = warp_slice_start; i + lane < warp_slice_end; i += 32) {
    T const item = data[i + lane];
    u32 const hash = hasher(item);
    u16 const digit = (hash >> place * RADIX_SIZE) & RADIX_MASK;

    u32 const warp_usage = min(warp_slice_end - i, 32);
    u32 const thread_mask = __funnelshift_lc(UINT_MAX, 0, warp_usage);

    u32 const matches = match_any_polyfill(thread_mask, digit);
    u8 const leader_id = __ffs(matches);

    u16 exclusive_sum = 0;

    if (lane == leader_id) {
      exclusive_sum = counters[warp_id][digit];
      counters[warp_id][digit] = exclusive_sum + __popc(matches);
    }
    __syncwarp(thread_mask);

    // share the exclusive sum within the warp
    exclusive_sum = __shfl_sync(matches, exclusive_sum, leader_id);
    // each thread sums its own prefix
    exclusive_sum += __popc(matches & (1 << lane) - 1);

    // warp-relative prefixes are stored in shared memory
    exclusive_sums[(i % warp_slice_length) / 32] = exclusive_sum;
  }

  __syncthreads();

  u16 const subwarp_id = threadIdx.x / subwarp_size;
  u8 const subwarp_lane = threadIdx.x % subwarp_size;

  // iterate over the warp counters of each digit
  for (u16 i = subwarp_id; i < RADIX; i += subwarps) {
    u16 val = subwarp_lane < warp_slices ? counters[subwarp_lane][i] : 0;
    val = warp_inclusive_multiscan<subwarp_size>(val);
    if (subwarp_lane < warp_slices) {
      counters[subwarp_lane][i] = val;
    }
  }

  __syncthreads();

  // the paper explicitly specifies "RADIX arrays of slices counters"
  using dynamic_2d_extents = cuda::std::extents<u32, RADIX, cuda::std::dynamic_extent>;
  cuda::std::mdspan global_status_counters(linearized_global_status_counters, dynamic_2d_extents{gridDim.x});
  shared u16 offsets[RADIX]; // NOLINT(bugprone-dynamic-static-initializers)

  // shortcut to the block totals
  cuda::std::span const totals(counters[warp_slices - 1], RADIX);
  // chained scan exclusive sums
  shared u32 prefix_totals[RADIX]; // NOLINT(bugprone-dynamic-static-initializers)

  if (threadIdx.x < RADIX) {
    cuda::atomic_ref<StatusCounter, cuda::thread_scope_device> const global_status_counter(
        global_status_counters(threadIdx.x, block_id));

    // publish the block totals
    global_status_counter.store({block_id == 0 ? Status::Prefix : Status::Count, totals[threadIdx.x]},
                                cuda::memory_order_release);

    prefix_totals[threadIdx.x] = 0;
    if (block_id != 0) {
      for (u32 i = block_id - 1;; --i) {
        cuda::atomic_ref<StatusCounter, cuda::thread_scope_device> const sc_ref(global_status_counters(threadIdx.x, i));
        StatusCounter sc; // NOLINT(cppcoreguidelines-pro-type-member-init)
        do {
          sc = sc_ref.load(cuda::memory_order_acquire);
        } while (sc.status == Status::Invalid);

        prefix_totals[threadIdx.x] += sc.counter;
        if (sc.status == Status::Prefix)
          break;
      }

      // publish the total inclusive prefix
      global_status_counter.store({Status::Prefix, totals[threadIdx.x] + prefix_totals[threadIdx.x]},
                                  cuda::memory_order_release);
    }

    // scan the block totals
    offsets[threadIdx.x] = radix_scan(totals[threadIdx.x]);
  }

  __syncthreads();

  shared T bins[slice_length]; // NOLINT(bugprone-dynamic-static-initializers)
  // shared u32 destinations[slice_length]; // NOLINT(bugprone-dynamic-static-initializers)
}
} // namespace sorting
