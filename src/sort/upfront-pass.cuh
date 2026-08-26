#pragma once

#include "sort-common.cuh"

template <typename T, typename Hasher>
  requires is_hasher<Hasher, T>
kernel void upfront_count(T *const data, u32 const length, u32 (*const counters)[DIGIT_PLACES][RADIX], Hasher hasher) {

  // block-local counters
  shared u32 local_counters[DIGIT_PLACES][RADIX]; // NOLINT(bugprone-dynamic-static-initializers)
  auto &linearized_local_counters = *reinterpret_cast<u32(*)[ARRAY_LENGTH_2D(local_counters)]>(local_counters);

  for (u16 i = threadIdx.x; i < ARRAY_LENGTH(linearized_local_counters); i += blockDim.x) {
    linearized_local_counters[i] = 0;
  }

  // sync all threads after initialization
  __syncthreads();

  /* we use a block-shared counter array to reduce local memory usage,
  this requires atomic writes, but allows wider radices. This reduces
  the total number of sorting iterations, making up for the atomics.

  we iterate over adjacent points to coalesce global memory reads,
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
      for (u8 place = 0; place < DIGIT_PLACES; ++place) {
        // extract digit
        u16 const digit = (hash >> place * RADIX_SIZE) & RADIX_MASK;
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
  for (u16 i = threadIdx.x; i < ARRAY_LENGTH(linearized_global_counters); i += blockDim.x) {
    u32 const count = linearized_local_counters[i];
    if (count) {
      atomicAdd(&(linearized_global_counters)[i], count);
    }
  }
}

kernel void exclusive_scan(Counters *const counters, u32 const length) {
  auto result = radix_scan<true>((counters->counters)[blockIdx.x][threadIdx.x], length);
  if (threadIdx.x == 0) {
    (counters->degenerates)[blockIdx.x] = !result.has_value();
  }
  if (result) {
    (counters->counters)[blockIdx.x][threadIdx.x] = result.value();
  }
}
