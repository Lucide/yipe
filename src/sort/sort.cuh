#pragma once

#include <driver_types.h>
#include "onesweep.cuh"
#include "upfront-pass.cuh"

template <typename Data, typename Hasher>
  requires is_hasher<Hasher, Data>
void sort(GPUInfo const *const info, Data *const data, u32 const length, const Hasher hasher) {
  using Key = Key<Hasher, Data>;
  Counters<Key> *counters;

  handle_cuda_result(cudaMallocAsync(&counters, sizeof(*counters), cudaStreamPerThread),
                     "Sort: Allocate global counters");
  handle_cuda_error_debug("Sort: Allocate global counters");
  handle_cuda_result(cudaMemsetAsync(counters, 0, sizeof(*counters), cudaStreamPerThread),
                     "Sort: Memset global counters");
  handle_cuda_error_debug("Sort: Memset global counters");

  u32 const blocks_per_grid =
      min(info->multiprocessors * upfront::blocks_per_sm, ceil_div<u32, u16, u32>(length, upfront::slice_length));

  upfront::count<<<blocks_per_grid, upfront::threads_per_block>>>(data, length, counters->counters, hasher);
  handle_cuda_error_debug("Sort: `upfront_count` kernel");
  upfront::exclusive_scan<Key><<<Key::digit_places, RADIX>>>(counters, length);
  handle_cuda_error_debug("Sort: `exclusive_scan` kernel");

  u32 *atomic_counter;
  handle_cuda_result(cudaMallocAsync(&atomic_counter, sizeof(u32), cudaStreamPerThread), "Sort: Allocate id counter");
  handle_cuda_error_debug("Sort: Allocate id counter");
  handle_cuda_result(cudaMemsetAsync(atomic_counter, 0, sizeof(u32)), "Sort: Memset id counter");
  handle_cuda_error_debug("Sort: Memset id counter");
  u32 const slices = ceil_div<u32, u16, u32>(length, onesweep::slice_max_length);
  u32 const state_couters_size = RADIX * slices * sizeof(onesweep::StateCounter);
  onesweep::StateCounter *state_counters;
  handle_cuda_result(cudaMallocAsync(&state_counters, state_couters_size, cudaStreamPerThread),
                     "Sort: Allocate state/counters");
  handle_cuda_error_debug("Sort: Allocate state/counters");
  Data *dst;
  handle_cuda_result(cudaMallocAsync(&dst, length * sizeof(Data), cudaStreamPerThread), "Sort: Allocate helper array");
  handle_cuda_error_debug("Sort: Allocate helper array");
  Data *src = data;

  decltype(counters->degenerates) degenerates;
  handle_cuda_result(cudaMemcpy(degenerates, counters->degenerates, sizeof(degenerates), cudaMemcpyDeviceToHost),
                     "Sort: Download degenerate blocks");
  // onesweep iterations
  for (u8 place = 0; place < Key::digit_places; ++place) {
    if (degenerates[place])
      continue;
    // printf("place: %d\n", place);
    handle_cuda_result(cudaMemsetAsync(state_counters, 0, state_couters_size), "Sort: Memset state/counters");
    onesweep::onesweep<<<slices, onesweep::threads_per_block>>>(atomic_counter, src, dst, length, place, state_counters,
                                                                counters->counters[place], hasher);
    handle_cuda_error_debug("Sort: `onesweep` kernel");
    /*
        if (place == 0) {
          Data *temp = reinterpret_cast<Data *>(malloc(length * sizeof(Data)));
          if (temp == nullptr)
            printf("out of memory\n");
          handle_cuda_result(cudaMemcpy(temp, dst, length * sizeof(Data), cudaMemcpyDeviceToHost), "allocate print
       points"); printf("\n printing sorted digits\n"); for (u32 i = 0; i < length && length <= 4096; ++i) { typename
       Key::KeyType hash = hasher(temp[i]); printf("%d ", Key::extract_digit(hash, 0));
          }
          printf("\n");
          free(temp);
        }
    */
    std::swap(src, dst);
  }

  if (dst == data) {
    handle_cuda_result(cudaMemcpyAsync((void *)data, (void *)src, length * sizeof(Data), cudaMemcpyDeviceToDevice),
                       "Sort: Memcpy sorted points");
    handle_cuda_error_debug("Sort: Memcpy sorted points");
    handle_cuda_result(cudaFreeAsync((void *)src, cudaStreamPerThread), "Sort: Free helper array");
    handle_cuda_error_debug("Sort: Free helper array");
  } else {
    handle_cuda_result(cudaFreeAsync((void *)dst, cudaStreamPerThread), "Sort: Free helper array");
    handle_cuda_error_debug("Sort: Free helper array");
  }
  /*
    for (u32 place = 0; place < ARRAY_LENGTH(degenerates); ++place) {
      printf("%d, ", degenerates[place]);
      // if (degenerates[place]
    }
    printf("\n");
    // debug copyback
    auto cpu_counters = new u32[1][Key::digit_places][RADIX];
    cudaMemcpy(cpu_counters, counters->counters, sizeof(counters->counters), cudaMemcpyDeviceToHost);
    for (u32 place = 0; place < Key::digit_places; ++place) {
      if (degenerates[place])
        continue;
      printf("place %d: [", place);
      for (u32 digit = 0; digit < RADIX; ++digit) {
        printf("%d:%d,", digit, (*cpu_counters)[place][digit]);
      }
      printf("]\n");
    }
    delete[] cpu_counters;
    // end debug copyback
   */
  handle_cuda_result(cudaFreeAsync(counters, cudaStreamPerThread), "Sort: Free counters");
  handle_cuda_error_debug("Sort: Free counters");
  handle_cuda_result(cudaFreeAsync(atomic_counter, cudaStreamPerThread), "Sort: Free id counter");
  handle_cuda_error_debug("Sort: Free id counter");
  handle_cuda_result(cudaFreeAsync(state_counters, cudaStreamPerThread), "Sort: Free state/counters");
  handle_cuda_error_debug("Sort: Free state/counters");
}
