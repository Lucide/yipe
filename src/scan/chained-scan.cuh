#pragma once

#include <cuda/atomic>
#include <cuda/std/mdspan>
#include "scan-common.cuh"

namespace chained_scan {
inline constexpr u16 threads_per_block = 1024;
static_assert(threads_per_block % 32 == 0, "threads_per_block must be a multiple of 32");

inline constexpr u16 items_per_thread = 12;
} // namespace chained_scan

namespace chained_scan {
inline constexpr u16 slice_length = threads_per_block * items_per_thread;
inline constexpr u8 warp_slices = threads_per_block / 32;
static_assert(warp_slices <= 32, "warp_slices must be at most 32");
inline constexpr u16 warp_slice_length = 32 * items_per_thread;
// inline constexpr u8 warp_slices_ceil = pow2_ceiling(warp_slices);

template <std::integral Data>
kernel void
chained_scan(u32 *const atomic_id, Data *const data, u32 const length, StatusCounter *const global_status_counters) {
  shared u32 block_id; // NOLINT(bugprone-dynamic-static-initializers)
  if (threadIdx.x == 0) {
    block_id = atomicAdd(atomic_id, 1) % gridDim.x;
  }

  // RADIX is multiple of 32, thus row reads are guaranteed to conflict
  // RADIX + subwarps_per_warp offsets the rows by subwarp slice
  shared Data slice_prefixes[warp_slices]; // NOLINT(bugprone-dynamic-static-initializers)
  for (u16 i = threadIdx.x; i < size(slice_prefixes); i += blockDim.x) {
    slice_prefixes[i] = 0;
  }

  __syncthreads();

  u8 const lane = threadIdx.x % 32;
  u16 const warp_id = threadIdx.x / 32;
  u32 const slice_start = block_id * slice_length;
  u32 const warp_slice_start = slice_start + warp_id * warp_slice_length;
  u32 const warp_slice_end = min(length, warp_slice_start + warp_slice_length);

  Data warp_slice_prefix[items_per_thread]; // NOLINT(bugprone-dynamic-static-initializers)

  // iterate over the data slice, partitioned by warps
  for (u32 i = lane; i < warp_slice_end - warp_slice_start; i += 32) {
    u16 const pass = i / 32;
    u32 const warp_usage = min(warp_slice_end - warp_slice_start - i - lane, 32);
    Data const item = data[i];
    u32 const thread_mask = __funnelshift_lc(UINT_MAX, 0, warp_usage);
    u8 const leader_id = warp_usage - 1;
    Data prefix = warp_inclusive_multiscan(item);

    Data sum = 0;

    if (lane == leader_id) {
      sum = slice_prefixes[warp_id];
      slice_prefixes[warp_id] = sum + prefix;
    }
    __syncwarp(thread_mask);

    // share the exclusive sum within the warp
    prefix = __shfl_sync(thread_mask, sum, leader_id);

    warp_slice_prefix[pass] = sum + prefix - item;
  }

  __syncthreads();

  if (warp_id == 0) {
    Data val = lane < warp_slices ? slice_prefixes[threadIdx.x] : 0;
    val = warp_inclusive_multiscan<pow2_ceiling(warp_slices)>(val);
    if (lane < warp_slices) {
      slice_prefixes[threadIdx.x] = val;
    }
  }

  __syncthreads();

  // chained scan exclusive sums
  shared Data global_prefix; // NOLINT(bugprone-dynamic-static-initializers)

  if (threadIdx.x == 0) {
    Data const &slice_total = slice_prefixes[warp_slices - 1];
    // fenced global memory status_counter
    cuda::atomic_ref<StatusCounter, cuda::thread_scope_device> const global_status_counter(
        global_status_counters[block_id]);

    // publish the block totals
    global_status_counter.store({block_id == 0 ? Status::Prefix : Status::Count, slice_total},
                                cuda::memory_order_release);

    global_prefix = 0;
    for (u32 i = block_id - 1; block_id != 0; --i) {
      cuda::atomic_ref<StatusCounter, cuda::thread_scope_device> const sc_ref(global_status_counters[i]);
      StatusCounter sc; // NOLINT(cppcoreguidelines-pro-type-member-init)
      do {
        sc = sc_ref.load(cuda::memory_order_acquire);
      } while (sc.status == Status::Invalid);

      global_prefix += sc.counter;
      if (sc.status == Status::Prefix) {
        // publish the total inclusive prefix
        global_status_counter.store({Status::Prefix, global_prefix[threadIdx.x] + slice_total},
                                    cuda::memory_order_release);
        break;
      }
    }
  }

  __syncthreads();

  for (u32 i = lane; i < warp_slice_end - warp_slice_start; i += 32) {
    u16 const pass = i / 32;
    data[i] = global_prefix + (warp_id == 0 ? 0 : slice_prefixes[warp_id - 1]) + warp_slice_prefix[pass];
  }
}
} // namespace chained_scan
