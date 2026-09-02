#pragma once

#include <cuda/atomic>
#include "scan-common.cuh"

namespace chained_scan {
enum class State : u8 {
  Invalid = 0,
  Count = 1,
  Prefix = 2,
};

inline constexpr u16 threads_per_block = 1024;
static_assert(threads_per_block % 32 == 0, "threads_per_block must be a multiple of 32");

inline constexpr u16 items_per_thread = 12;
} // namespace chained_scan

namespace chained_scan {
inline constexpr u16 slice_max_length = threads_per_block * items_per_thread;
inline constexpr u8 warp_slices = threads_per_block / 32;
static_assert(warp_slices <= 32, "warp_slices must be at most 32");
inline constexpr u16 warp_slice_max_length = 32 * items_per_thread;

template <std::integral Data>
KERNEL void chained_scan(u32 *const atomic_id,
                         Data *const data,
                         u32 const length,
                         chained_scan::State *global_states,
                         Data *linearized_global_counters_prefixes) {
  SHARED u32 block_id; // NOLINT(bugprone-dynamic-static-initializers)
  if (threadIdx.x == 0) {
    block_id = atomicAdd(atomic_id, 1) % gridDim.x;
  }

  // RADIX is multiple of 32, thus row reads are guaranteed to conflict
  // RADIX + subwarps_per_warp offsets the rows by subwarp slice
  SHARED Data slice_prefixes[warp_slices]; // NOLINT(bugprone-dynamic-static-initializers)
  for (u16 i = threadIdx.x; i < cuda::std::size(slice_prefixes); i += blockDim.x) {
    slice_prefixes[i] = 0;
  }

  __syncthreads();

  const cuda::std::span<Data, cuda::std::dynamic_extent> global_counters(linearized_global_counters_prefixes,
                                                                         gridDim.x);
  const cuda::std::span<Data, cuda::std::dynamic_extent> global_prefixes(
      &linearized_global_counters_prefixes[gridDim.x], gridDim.x);
  u8 const lane = threadIdx.x % 32;
  u16 const warp_id = threadIdx.x / 32;
  u32 const slice_start = block_id * slice_max_length;
  u32 const warp_slice_start = min(length, slice_start + warp_id * warp_slice_max_length);
  u32 const warp_slice_end = min(length, warp_slice_start + warp_slice_max_length);
  u16 const warp_slice_length = warp_slice_start - warp_slice_end;

  Data warp_slice_prefix[items_per_thread]; // NOLINT(bugprone-dynamic-static-initializers)

  // iterate over the data slice, partitioned by warps
  for (u32 i = lane; i < warp_slice_length; i += 32) {
    u16 const pass = i / 32;
    u8 const warp_usage = min(warp_slice_length - i - lane, 32);
    Data const item = data[warp_slice_start + i];
    u32 const thread_mask = __funnelshift_lc(UINT_MAX, 0, warp_usage);
    u8 const leader_id = warp_usage - 1;
    Data warp_prefix = warp_inclusive_multiscan(item);

    Data prefix = 0;

    if (lane == leader_id) {
      prefix = slice_prefixes[warp_id];
      slice_prefixes[warp_id] += warp_prefix;
    }
    __syncwarp(thread_mask);

    // share the exclusive sum within the warp
    warp_prefix = __shfl_sync(thread_mask, prefix, leader_id);
    // warp_slice-relative sums are stored in shared memory
    warp_slice_prefix[pass] = prefix + warp_prefix - item;
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
  SHARED Data global_prefix; // NOLINT(bugprone-dynamic-static-initializers)

  if (threadIdx.x == 0) {
    Data const &slice_total = slice_prefixes[warp_slices - 1];
    // fenced global memory status_counter
    cuda::atomic_ref<State, cuda::thread_scope_device> const global_state(global_states[block_id]);

    // publish the block totals
    if (block_id == 0) {
      global_prefixes[block_id] = slice_total;
    } else {
      global_counters[block_id] = slice_total;
    }
    global_state.store(block_id == 0 ? State::Prefix : State::Count, cuda::memory_order_release);

    global_prefix = 0;
    for (u32 i = block_id - 1; block_id != 0; --i) {
      cuda::atomic_ref<State, cuda::thread_scope_device> const sc_ref(global_states[i]);
      State sc = sc_ref.load(cuda::memory_order_acquire);
      while (sc == State::Invalid) {
        nanosleep_polyfill(32);
        sc = sc_ref.load(cuda::memory_order_acquire);
      }

      if (sc == State::Prefix) {
        // publish the total inclusive prefix
        global_prefix += global_prefixes[i];
        global_prefixes[i] = global_prefix + slice_total;
        global_state.store(State::Prefix, cuda::memory_order_release);
        break;
      }
      global_prefix += global_counters[i];
    }
  }

  __syncthreads();

  for (u32 i = lane; i < warp_slice_length; i += 32) {
    u16 const pass = i / 32;
    data[i] = global_prefix + (warp_id == 0 ? 0 : slice_prefixes[warp_id - 1]) + warp_slice_prefix[pass];
  }
}
} // namespace chained_scan
