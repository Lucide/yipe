#pragma once

#include "sort/onesweep.cuh"
#include "sort/upfront-pass.cuh"
#include <cstdio>

template <typename T, typename Hasher>
void sort(GPUInfo const *const info, T *const data, u32 const length, Hasher hasher) {
  u32(*counters)[DIGIT_PLACES][RADIX];
  cudaMalloc(&counters, sizeof(*counters));
  cudaMemset(counters, 0, sizeof(*counters));

  u32 const blocks_per_grid = min(info->multiprocessors * counting::blocks_per_sm,
                                  (length + counting::slice_length - 1) / counting::slice_length);
  printf(
      "info:\n\tblocks per grid: %d\n\tthreads per block: %d\n\tmultiprocessors: %d\n\tcores per multiprocessor: %d\n",
      blocks_per_grid, counting::threads_per_block, info->multiprocessors, info->cores_per_multiprocessor);

  upfront_count<<<blocks_per_grid, counting::threads_per_block>>>(data, length, counters, hasher);
  handle_cuda_error();
  exclusive_scan<<<DIGIT_PLACES, RADIX>>>(counters);
  handle_cuda_error();
  u32 *atomic_counter;
  cudaMalloc(&atomic_counter, sizeof(u32));
  cudaMemset(atomic_counter, 0, sizeof(u32));

  u32 const slices = (length + sorting::slice_length - 1) / sorting::slice_length;
  u32 *status_counters;
  cudaMalloc(&status_counters, RADIX * slices);

  // onesweep iterations
  for (u32 place = 0; place < ARRAY_LENGTH(*counters); ++place) {
    onesweep<<<slices, sorting::threads_per_block>>>(atomic_counter, data, length, place, status_counters,
                                                     counters[place], hasher);
  }

  // debug copyback
  auto cpu_counters = new u32[1][DIGIT_PLACES][RADIX];
  cudaMemcpy(cpu_counters, counters, sizeof(u32[DIGIT_PLACES][RADIX]), cudaMemcpyDeviceToHost);
  printf("sorted\n");
  for (u32 i = 0; i < DIGIT_PLACES; ++i) {
    printf("place %d: [", i);
    for (u32 j = 0; j < RADIX; ++j) {
      printf("%d:%d,", j, (*cpu_counters)[i][j]);
    }
    printf("]\n");
  }
  delete[] cpu_counters;
  // end debug copyback

  cudaFree(counters);
  cudaFree(atomic_counter);
  cudaFree(status_counters);
}
