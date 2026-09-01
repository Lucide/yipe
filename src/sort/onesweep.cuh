#pragma once

#include <cuda/atomic>
#include <cuda/std/mdspan>
#include "sort-common.cuh"

namespace onesweep {
inline constexpr u16 slice_length = threads_per_block * items_per_thread;
inline constexpr u8 warp_slices = threads_per_block / 32;
static_assert(warp_slices <= 32, "warp_slices must be at most 32");
inline constexpr u16 warp_slice_length = 32 * items_per_thread;
inline constexpr u8 subwarp_size = pow2_ceiling(warp_slices);
inline constexpr u8 subwarps_per_warp = 32 / subwarp_size;
inline constexpr u16 subwarps = warp_slices * subwarps_per_warp;

template <typename Data, typename Hasher>
  requires is_hasher<Hasher, Data>
kernel void onesweep(u32 *const atomic_id,
                     const Data *const data,
                     Data *const sorted,
                     u32 const length,
                     u8 const place,
                     StatusCounter *const linearized_global_status_counters,
                     u32 (&global_totals)[RADIX],
                     Hasher hasher) {
  using Key = Key<Hasher, Data>;
  shared u32 block_id; // NOLINT(bugprone-dynamic-static-initializers)
  if (threadIdx.x == 0) {
    block_id = atomicAdd(atomic_id, 1) % gridDim.x;
  }

  // RADIX is multiple of 32, thus row reads are guaranteed to conflict
  // RADIX + subwarps_per_warp offsets the rows by subwarp slice
  shared u16 slice_prefixes[warp_slices][RADIX + subwarps_per_warp]; // NOLINT(bugprone-dynamic-static-initializers)
  auto linearized_slice_prefixes = flatten(slice_prefixes);
  for (u16 i = threadIdx.x; i < size(linearized_slice_prefixes); i += blockDim.x) {
    linearized_slice_prefixes[i] = 0;
  }

  __syncthreads();

  u8 const lane = threadIdx.x % 32;
  u16 const warp_id = threadIdx.x / 32;
  u16 const subwarp_id = threadIdx.x / subwarp_size;
  u8 const subwarp_lane = threadIdx.x % subwarp_size;
  u32 const slice_start = block_id * slice_length;
  u32 const slice_end = min(length, slice_start + slice_length);
  u32 const warp_slice_start = slice_start + warp_id * warp_slice_length;
  u32 const warp_slice_end = min(length, warp_slice_start + warp_slice_length);

  u16 warp_slice_prefix[items_per_thread]; // NOLINT(bugprone-dynamic-static-initializers)

  // iterate over the data slice, partitioned by warps
  for (u32 i = lane; i < warp_slice_end - warp_slice_start; i += 32) {
    u16 const pass = i / 32;
    u32 const warp_usage = min(warp_slice_end - warp_slice_start - i - lane, 32);
    u32 const thread_mask = __funnelshift_lc(UINT_MAX, 0, warp_usage);
    Data const item = data[i];
    typename Key::KeyType const hash = hasher(item);
    u16 const digit = Key::extract_digit(hash, place);
    u32 const matches = match_any_polyfill(thread_mask, digit);
    // __ffs is 1-indexed
    u8 const leader_id = __ffs(matches) - 1;

    u16 sum = 0;

    if (lane == leader_id) {
      sum = slice_prefixes[warp_id][digit];
      slice_prefixes[warp_id][digit] = sum + __popc(matches);
    }
    __syncwarp(thread_mask);

    // share the exclusive sum within the warp
    sum = __shfl_sync(thread_mask, sum, leader_id);

    // each thread sums its own prefix
    // warp_slice-relative counts are stored in shared memory
    warp_slice_prefix[pass] = sum + __popc(matches & (1u << lane) - 1);
  }

  __syncthreads();

  // iterate over the warp_slice counters of each digit
  for (u16 i = subwarp_id; i < RADIX; i += subwarps) {
    u16 val = subwarp_lane < warp_slices ? slice_prefixes[subwarp_lane][i] : 0;
    val = warp_inclusive_multiscan<subwarp_size>(val);
    if (subwarp_lane < warp_slices) {
      slice_prefixes[subwarp_lane][i] = val;
    }
  }

  __syncthreads();

  // chained scan exclusive sums
  shared u32 global_prefix[RADIX]; // NOLINT(bugprone-dynamic-static-initializers)
  cuda::std::span const slice_totals(slice_prefixes[warp_slices - 1], RADIX);

  if (threadIdx.x < RADIX) {
    // slice containing the slice totals
    // the paper explicitly specifies "RADIX arrays of slices counters"
    using global_status_counters_extents = cuda::std::extents<u32, RADIX, cuda::std::dynamic_extent>;
    cuda::std::mdspan global_status_counters(linearized_global_status_counters,
                                             global_status_counters_extents{gridDim.x});
    // fenced global memory status_counter
    cuda::atomic_ref<StatusCounter, cuda::thread_scope_device> const global_status_counter(
        global_status_counters(threadIdx.x, block_id));

    // publish the block totals
    global_status_counter.store({block_id == 0 ? Status::Prefix : Status::Count, slice_totals[threadIdx.x]},
                                cuda::memory_order_release);

    global_prefix[threadIdx.x] = 0;
    for (u32 i = block_id - 1; block_id != 0; --i) {
      cuda::atomic_ref<StatusCounter, cuda::thread_scope_device> const sc_ref(global_status_counters(threadIdx.x, i));
      StatusCounter sc = sc_ref.load(cuda::memory_order_acquire);
      while (sc.status == Status::Invalid) {
        nanosleep_polyfill(32);
        sc = sc_ref.load(cuda::memory_order_acquire);
      }

      global_prefix[threadIdx.x] += sc.counter;
      if (sc.status == Status::Prefix) {
        // publish the total inclusive prefix
        global_status_counter.store({Status::Prefix, global_prefix[threadIdx.x] + slice_totals[threadIdx.x]},
                                    cuda::memory_order_release);
        break;
      }
    }

    // scan the block totals
    slice_totals[threadIdx.x] = radix_scan(slice_totals[threadIdx.x]);
  }

  __syncthreads();

  shared Data bins[slice_length]; // NOLINT(bugprone-dynamic-static-initializers)

  // iterate over the data slice, partitioned by warps
  for (u32 i = warp_slice_start; i + lane < warp_slice_end; i += 32) {
    u16 const pass = (i - warp_slice_start) / 32;

    Data const item = data[i + lane];
    typename Key::KeyType const hash = hasher(item);
    u16 const digit = Key::extract_digit(hash, place);

    u16 const dest =
        slice_totals[digit] + (warp_id == 0 ? 0 : slice_prefixes[warp_id - 1][digit]) + warp_slice_prefix[pass];
    bins[dest] = item;
  }

  __syncthreads();

  for (u16 i = threadIdx.x; i < slice_end - slice_start; i += blockDim.x) {
    Data const item = bins[i];
    typename Key::KeyType const hash = hasher(item);
    u16 const digit = Key::extract_digit(hash, place);

    u32 const global_dest = global_totals[digit] + global_prefix[digit] + i - slice_totals[digit];
    sorted[global_dest] = item;
  }
}
} // namespace onesweep
