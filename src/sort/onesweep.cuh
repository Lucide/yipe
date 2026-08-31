#pragma once

#include <cuda/atomic>
#include <cuda/std/mdspan>
#include "sort-common.cuh"

// #define __CUDA_ARCH__ 700
// #define __CUDA_ARCH__ 610
namespace onesweep {
enum class Status : u32 {
  Invalid = 0,
  Count = 1,
  Prefix = 2,
};

struct StatusCounter {
  Status status : 2;
  u32 counter : 30;
};

static_assert(sizeof(StatusCounter) == sizeof(u32), "status and counter must be packed in a single 32 bit word");

inline constexpr u16 slice_length = threads_per_block * items_per_thread;
inline constexpr u8 warp_slices = threads_per_block / 32;
static_assert(warp_slices <= 32, "warp_slices must be at most 32");
inline constexpr u16 warp_slice_length = 32 * items_per_thread;
// (32 - count_leading_zeroes(n-1)) calculates the smallest power of two >=n
// undefined behaviour for clz(0)
inline constexpr u8 subwarp_size = (warp_slices == 1) ? 1u : 1u << (32u - std::countl_zero(warp_slices - 1u));
inline constexpr u8 subwarps_per_warp = 32 / subwarp_size;
inline constexpr u16 subwarps = warp_slices * subwarps_per_warp;

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
                     const T *const data,
                     T *const sorted,
                     u32 const length,
                     u8 const place,
                     StatusCounter *const linearized_global_status_counters,
                     u32 (&global_totals)[RADIX],
                     Hasher hasher) {
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
  for (u32 i = warp_slice_start; i + lane < warp_slice_end; i += 32) {
    u16 const pass = (i - warp_slice_start) / 32;

    T const item = data[i + lane];
    u32 const hash = hasher(item);
    u16 const digit = (hash >> place * RADIX_SIZE) & RADIX_MASK;

    u32 const warp_usage = min(warp_slice_end - i, 32);
    u32 const thread_mask = __funnelshift_lc(UINT_MAX, 0, warp_usage);
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
  cuda::std::span const slice_counters(slice_prefixes[warp_slices - 1], RADIX);

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
    global_status_counter.store({block_id == 0 ? Status::Prefix : Status::Count, slice_counters[threadIdx.x]},
                                cuda::memory_order_release);

    global_prefix[threadIdx.x] = 0;
    for (u32 i = block_id - 1; block_id != 0; --i) {
      cuda::atomic_ref<StatusCounter, cuda::thread_scope_device> const sc_ref(global_status_counters(threadIdx.x, i));
      StatusCounter sc; // NOLINT(cppcoreguidelines-pro-type-member-init)
      do {
        sc = sc_ref.load(cuda::memory_order_acquire);
      } while (sc.status == Status::Invalid);

      global_prefix[threadIdx.x] += sc.counter;
      if (sc.status == Status::Prefix) {
        // publish the total inclusive prefix
        global_status_counter.store({Status::Prefix, global_prefix[threadIdx.x] + slice_counters[threadIdx.x]},
                                    cuda::memory_order_release);
        break;
      }
    }

    // scan the block totals
    slice_counters[threadIdx.x] = radix_scan(slice_counters[threadIdx.x]);
  }

  __syncthreads();

  shared T bins[slice_length]; // NOLINT(bugprone-dynamic-static-initializers)

  // iterate over the data slice, partitioned by warps
  for (u32 i = warp_slice_start; i + lane < warp_slice_end; i += 32) {
    u16 const pass = (i - warp_slice_start) / 32;

    T const item = data[i + lane];
    u32 const hash = hasher(item);
    u16 const digit = (hash >> place * RADIX_SIZE) & RADIX_MASK;

    u16 const dest =
        slice_counters[digit] + (warp_id == 0 ? 0 : slice_prefixes[warp_id - 1][digit]) + warp_slice_prefix[pass];
    bins[dest] = item;
  }

  __syncthreads();

  for (u16 i = threadIdx.x; i < slice_end - slice_start; i += blockDim.x) {
    T const item = bins[i];
    u32 const hash = hasher(item);
    u16 const digit = (hash >> place * RADIX_SIZE) & RADIX_MASK;

    u32 const global_dest = global_totals[digit] + global_prefix[digit] + (i - slice_counters[digit]);
    sorted[global_dest] = item;

    if (i < ARRAY_LENGTH(bins) - 1) {
      T const item2 = bins[i + 1];
      u32 const hash2 = hasher(item);
      u16 const partial = hash & __funnelshift_lc(UINT_MAX, RADIX_MASK, digit * 8);
      u16 const partial2 = hash2 & __funnelshift_lc(UINT_MAX, RADIX_MASK, digit * 8);
      if (partial > partial2) {
        printf("%d %d\n", partial, partial2);
        printf("error\n");
      }
    }
  }
}
} // namespace onesweep
