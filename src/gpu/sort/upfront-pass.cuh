#pragma once

#include "sort-common.cuh"

namespace upfront {
inline constexpr u32 slice_length = threads_per_block * items_per_thread_per_slice;

template <typename Data, is_hasher<Data> Hasher>
KERNEL void count(Data const *const data,
                  u32 const length,
                  u32 (&counters)[Key<Data, Hasher>::digit_places][RADIX],
                  Hasher const hasher) {
  // block-local counters
  using Key = Key<Data, Hasher>;
  SHARED u32 local_counters[Key::digit_places][RADIX]; // NOLINT(bugprone-dynamic-static-initializers)
  auto const linearized_local_counters = flatten(local_counters);

  for (u16 i = static_cast<u16>(threadIdx.x); i < cuda::std::size(linearized_local_counters); i += blockDim.x) {
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
  for (u32 i = blockIdx.x * slice_length; i < length; i += gridDim.x * slice_length) {
    // iterate over the data slice in strides of block size
    u32 const slice_end = min(length, i + slice_length);
    for (u32 j = i + threadIdx.x; j < slice_end; j += blockDim.x) {
      Data const item = data[j];
      typename Key::KeyType const hash = hasher(item);
      // for each digit place in the key
      for (u8 place = 0; place < Key::digit_places; ++place) {
        // extract digit
        u16 const digit = Key::extract_digit(hash, place);
        // increment corresponding counter
        atomicAdd_block(&local_counters[place][digit], 1);
      }
    }
  }

  // sync all threads before starting write
  __syncthreads();

  // cast the arrays as a flat 1D layout
  auto const linearized_global_counters = flatten(counters);
  // parallel write to global memory
  for (u16 i = static_cast<u16>(threadIdx.x); i < cuda::std::size(linearized_global_counters); i += blockDim.x) {
    u32 const count = linearized_local_counters[i];
    if (count) {
      atomicAdd(&(linearized_global_counters)[i], count);
    }
  }
}

template <IsInstanceOf<Key> Key> KERNEL void exclusive_scan(Counters<Key> *const counters, u32 const length) {
  auto const result = radix_scan<true>((counters->counters)[blockIdx.x][threadIdx.x], length);
  if (threadIdx.x == 0) {
    (counters->degenerates)[blockIdx.x] = !result.has_value();
  }
  if (result) {
    (counters->counters)[blockIdx.x][threadIdx.x] = *result;
  }
}
} // namespace upfront
