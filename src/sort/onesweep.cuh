#pragma once

#include <cuda/atomic>
#include <cuda/std/mdspan>
#include "sort-common.cuh"

namespace onesweep {
enum class State : u32 {
  Invalid = 0,
  Count = 1,
  Prefix = 2,
};

struct StateCounter {
  State status : 2;
  u32 counter : 30;
};
static_assert(sizeof(StateCounter) == sizeof(u32), "status and counter must be packed in a single 32 bit word");

inline constexpr u16 slice_max_length = threads_per_block * items_per_thread;
inline constexpr u8 warp_slices = threads_per_block / 32;
static_assert(warp_slices <= 32, "warp_slices must be at most 32");
inline constexpr u16 warp_slice_max_length = 32 * items_per_thread;
inline constexpr u8 subwarp_size = pow2_ceiling(warp_slices);
inline constexpr u8 subwarps_per_warp = 32 / subwarp_size;
inline constexpr u16 subwarps = warp_slices * subwarps_per_warp;

template <typename Data, typename Hasher>
  requires is_hasher<Hasher, Data>
KERNEL void onesweep(u32 *const atomic_id,
                     Data const *const data,
                     Data *const sorted,
                     u32 const length,
                     u8 const place,
                     StateCounter *const linearized_global_state_counters,
                     u32 const (&global_totals)[RADIX],
                     Hasher const hasher) {
  using Key = Key<Hasher, Data>;
  SHARED u32 block_id; // NOLINT(bugprone-dynamic-static-initializers)
  if (threadIdx.x == 0) {
    block_id = atomicAdd(atomic_id, 1) % gridDim.x;
  }

  // RADIX is multiple of 32, thus row reads are guaranteed to conflict
  // RADIX + subwarps_per_warp offsets the rows by subwarp slice
  SHARED u16 slice_prefixes[warp_slices][RADIX + subwarps_per_warp]; // NOLINT(bugprone-dynamic-static-initializers)
  auto const linearized_slice_prefixes = flatten(slice_prefixes);
  for (u16 i = static_cast<u16>(threadIdx.x); i < static_cast<u16>(cuda::std::size(linearized_slice_prefixes));
       i += blockDim.x) {
    linearized_slice_prefixes[i] = 0;
  }

  __syncthreads();

  u8 const lane = threadIdx.x % 32;
  u16 const warp_id = static_cast<u16>(threadIdx.x) / 32;
  u16 const subwarp_id = static_cast<u16>(threadIdx.x) / subwarp_size;
  u8 const subwarp_lane = threadIdx.x % subwarp_size;
  u32 const slice_start = block_id * slice_max_length;
  u32 const slice_end = min(length, slice_start + slice_max_length);
  u16 const slice_length = static_cast<u16>(slice_end - slice_start);
  u32 const warp_slice_start = min(slice_end, slice_start + warp_id * warp_slice_max_length);
  u32 const warp_slice_end = min(slice_end, warp_slice_start + warp_slice_max_length);
  u16 const warp_slice_length = static_cast<u16>(warp_slice_end - warp_slice_start);

  u16 warp_slice_prefix[items_per_thread]; // NOLINT(bugprone-dynamic-static-initializers)

  // iterate over the data slice, partitioned by warps
  // iterate over `pass` with explicit bound instead of `i`
  // prevents spilling of `warp_slice_prefix`
  for (u16 pass = 0; pass < items_per_thread && pass * 32 + lane < warp_slice_length; ++pass) {
    u32 i = pass * 32 + lane;
    u8 const warp_usage = static_cast<u8>(min(warp_slice_length - pass * 32, 32U));
    u32 const thread_mask = __funnelshift_lc(UINT_MAX, 0, warp_usage);
    Data const item = data[warp_slice_start + i];
    typename Key::KeyType const hash = hasher(item);
    u16 const digit = Key::extract_digit(hash, place);
    u32 const matches = match_any_polyfill<RADIX_SIZE>(digit, thread_mask);
    // __ffs is 1-indexed
    u8 const leader_id = __ffs(matches) - 1;

    u16 prefix = 0;

    if (lane == leader_id) {
      prefix = slice_prefixes[warp_id][digit];
      slice_prefixes[warp_id][digit] += __popc(matches);
    }
    __syncwarp(thread_mask);

    // share the exclusive sum within the warp
    prefix = __shfl_sync(thread_mask, prefix, leader_id);
    // each thread sums its own prefix
    // warp_slice-relative counts are stored in shared memory
    warp_slice_prefix[pass] = prefix + __popc(matches & (1u << lane) - 1);
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
  SHARED u32 global_prefix[RADIX]; // NOLINT(bugprone-dynamic-static-initializers)
  cuda::std::span const slice_totals(slice_prefixes[warp_slices - 1], RADIX);

  if (threadIdx.x < RADIX) {
    // slice containing the slice totals
    // the paper explicitly specifies "RADIX arrays of slices counters"
    using global_state_counters_extents = cuda::std::extents<u32, RADIX, cuda::std::dynamic_extent>;
    cuda::std::mdspan const global_state_counters(linearized_global_state_counters,
                                                  global_state_counters_extents{gridDim.x});
    // fenced global memory state_counter
    cuda::atomic_ref<StateCounter, cuda::thread_scope_device> const global_state_counter(
        global_state_counters(threadIdx.x, block_id));

    // publish the block totals
    global_state_counter.store({block_id == 0 ? State::Prefix : State::Count, slice_totals[threadIdx.x]},
                               cuda::memory_order_release);

    global_prefix[threadIdx.x] = 0;
    for (u32 i = block_id - 1; block_id != 0; --i) {
      cuda::atomic_ref<StateCounter, cuda::thread_scope_device> const sc_ref(global_state_counters(threadIdx.x, i));
      StateCounter sc = sc_ref.load(cuda::memory_order_acquire);
      while (sc.status == State::Invalid) {
        nanosleep_polyfill(32);
        sc = sc_ref.load(cuda::memory_order_acquire);
      }

      global_prefix[threadIdx.x] += sc.counter;
      if (sc.status == State::Prefix) {
        // publish the total inclusive prefix
        global_state_counter.store({State::Prefix, global_prefix[threadIdx.x] + slice_totals[threadIdx.x]},
                                   cuda::memory_order_release);
        break;
      }
    }

    // scan the block totals
    slice_totals[threadIdx.x] = static_cast<u16>(radix_scan(slice_totals[threadIdx.x]));
  }

  __syncthreads();

  SHARED Data bins[slice_max_length]; // NOLINT(bugprone-dynamic-static-initializers)

  // iterate over the data slice, partitioned by warps
  // iterate over `pass` with explicit bound instead of `i`
  // prevents spilling of `warp_slice_prefix`
  for (u16 pass = 0; pass < items_per_thread && pass * 32 + lane < warp_slice_length; ++pass) {
    u32 i = pass * 32 + lane;

    Data const item = data[warp_slice_start + i];
    typename Key::KeyType const hash = hasher(item);
    u16 const digit = Key::extract_digit(hash, place);

    u16 const dest =
        slice_totals[digit] + (warp_id == 0 ? 0 : slice_prefixes[warp_id - 1][digit]) + warp_slice_prefix[pass];
    bins[dest] = item;
  }

  __syncthreads();

  for (u16 i = static_cast<u16>(threadIdx.x); i < slice_length; i += blockDim.x) {
    Data const item = bins[i];
    typename Key::KeyType const hash = hasher(item);
    u16 const digit = Key::extract_digit(hash, place);

    u32 const global_dest = global_totals[digit] + global_prefix[digit] + i - slice_totals[digit];
    sorted[global_dest] = item;
  }
}
} // namespace onesweep
