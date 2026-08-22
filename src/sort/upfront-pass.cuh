#pragma once

#include "sort-common.cuh"

global void exclusive_scan(u32 (*const counters)[DIGIT_PLACES][RADIX]) {
  constexpr u32 warps = RADIX / 32;

  // shared memory to store the aggregate sum of each warp
  shared u32 warp_sums[warps]; // NOLINT(bugprone-dynamic-static-initializers)

  u32 const lane = threadIdx.x % 32;
  u32 const warp_id = threadIdx.x / 32;

  // scan within each warp
  u32 val = warp_inclusive_multiscan<32>((*counters)[blockIdx.x][threadIdx.x]);

  // thread 31 in each warp publishes the warp total
  if (lane == 31) {
    warp_sums[warp_id] = val;
  }
  __syncthreads();

  // warp 0 scans the warp totals
  if (warp_id == 0) {
    u32 const warp_val = warp_inclusive_multiscan<warps>((lane < warps) ? warp_sums[lane] : 0);
    if (lane < warps) {
      warp_sums[lane] = warp_val;
    }
  }
  __syncthreads();

  val = __shfl_up_sync(SYNC_WARP, val, 1);
  if (lane == 0) {
    val = 0;
  }

  // add prefix sum of previous warps to local warp results
  if (warp_id > 0) {
    val += warp_sums[warp_id - 1];
  }

  (*counters)[blockIdx.x][threadIdx.x] = val;
}

template <typename T, typename Hasher>
  requires is_hasher<Hasher, T>
global void upfront_count(T *const data, u32 const length, u32 (*const counters)[DIGIT_PLACES][RADIX], Hasher hasher) {

  // block-local counters
  shared u32 local_counters[DIGIT_PLACES][RADIX]; // NOLINT(bugprone-dynamic-static-initializers)
  auto &linearized_local_counters = *reinterpret_cast<u32(*)[ARRAY_LENGTH_2D(local_counters)]>(local_counters);

  for (u32 i = threadIdx.x; i < DIGIT_PLACES * RADIX; i += blockDim.x) {
    linearized_local_counters[i] = 0;
  }

  // sync all threads after initialization
  __syncthreads();

  /* We use a block-shared counter array to reduce local memory usage,
  this requires atomic writes, but allows wider radices. This reduces
  the total number of sorting iterations, making up for the atomics.

  We iterate over adjacent points to coalesce global memory reads,
  but because we don't know where upfront the counter to be increased
  there is no point trying to align local memory writes.
  */

  // iterate over slices in strides of grid size
  for (u32 i = blockIdx.x * counting::slice_length; i < length; i += gridDim.x * counting::slice_length) {
    // iterate over the data slice in strides of block size
    u32 const slice_end = min(length, i + counting::slice_length);
    for (u32 j = i + threadIdx.x; j < slice_end; j += blockDim.x) {
      T const item = data[j];
      u32 const hash = hasher(item);
      // for each digit place in the key
      for (u32 place = 0; place < DIGIT_PLACES; ++place) {
        // extract digit
        u32 const digit = (hash >> place * RADIX_SIZE) & RADIX_MASK;
        // increment corresponding counter
        atomicAdd_block(&local_counters[place][digit], 1);
      }
    }
  }

  // sync all threads before starting write
  __syncthreads();

  // cast the arrays as a flat 1D layout
  auto &linearized_global_counters = *reinterpret_cast<u32(*)[ARRAY_LENGTH_2D(*counters)]>(*counters);
  // parallel write to global memory
  for (u32 i = threadIdx.x; i < ARRAY_LENGTH(linearized_global_counters); i += blockDim.x) {
    u32 const count = linearized_local_counters[i];
    if (count) {
      atomicAdd(&(linearized_global_counters)[i], count);
    }
  }
}
