#ifndef YIPE_UPFRONT_PASS_CUH
#define YIPE_UPFRONT_PASS_CUH

#include "common.h"
#include "gpu-common.cuh"

// intra-warp inclusive prefix sum using register shuffles
gpu inline u32 warp_inclusive_scan(u32 val) {
#pragma unroll
  for (u32 offset = 1; offset < 32; offset <<= 1) {
    u32 const remote_val = __shfl_up_sync(SYNC_WARP, val, offset);
    if ((threadIdx.x % 32) >= offset) {
      val += remote_val;
    }
  }
  return val;
}

global void exclusive_scan(u32 (*const counters)[DIGIT_PLACES][RADIX]) {
  // shared memory to store the aggregate sum of each of the 8 warps
  shared u32 warp_sums[8];

  u32 const lane_id = threadIdx.x % 32;
  u32 const warp_id = threadIdx.x / 32;

  // scan within each warp
  u32 val = warp_inclusive_scan((*counters)[blockIdx.x][threadIdx.x]);

  // thread 31 in each warp publishes the warp total
  if (lane_id == 31) {
    warp_sums[warp_id] = val;
  }
  __syncthreads();

  // warp 0 scans the 8 warp totals
  if (warp_id == 0) {
    u32 const warp_val = warp_inclusive_scan((lane_id < 8) ? warp_sums[lane_id] : 0);
    if (lane_id < 8) {
      warp_sums[lane_id] = warp_val;
    }
  }
  __syncthreads();

  val = __shfl_up_sync(SYNC_WARP, val, 1);
  if (lane_id == 0) {
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
global void upfront_count(T *const data, u32 const size, u32 (*const counters)[DIGIT_PLACES][RADIX], Hasher hasher) {

  // block-local counters
  shared u32 local_counters[DIGIT_PLACES][RADIX];
  auto(&linearized_local_counters)[DIGIT_PLACES * RADIX] =
      *reinterpret_cast<u32(*)[DIGIT_PLACES * RADIX]>(local_counters);

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
  for (u32 i = blockIdx.x * COUNTING_SLICE_SIZE; i < size; i += gridDim.x * COUNTING_SLICE_SIZE) {
    // iterate over the data slice in strides of block size
    u32 slice_end = min(size, i + COUNTING_SLICE_SIZE);
    for (u32 j = i + threadIdx.x; j < slice_end; j += blockDim.x) {
      T point = data[j];
      u32 hash = hasher(point);
      // for each digit place in the key
      for (u32 k = 0; k < DIGIT_PLACES; ++k) {
        // extract digit
        u32 digit = (hash & (RADIX_MASK << (k * RADIX_SIZE))) >> (k * RADIX_SIZE);
        // increment corresponding counter
        atomicAdd(&local_counters[k][digit], 1);
      }
    }
  }

  // sync all threads before starting write
  __syncthreads();

  // cast the arrays as a flat 1D layout
  auto(&linearized_global_counters)[DIGIT_PLACES * RADIX] = *reinterpret_cast<u32(*)[DIGIT_PLACES * RADIX]>(*counters);
  // parallel write to global memory
  for (u32 i = threadIdx.x; i < DIGIT_PLACES * RADIX; i += blockDim.x) {
    u32 count = linearized_local_counters[i];
    if (count) {
      atomicAdd(&(linearized_global_counters)[i], count);
    }
  }
}

#endif // YIPE_UPFRONT_PASS_CUH
