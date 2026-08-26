#pragma once

#include <cstdio>
#include "sort/onesweep.cuh"
#include "sort/upfront-pass.cuh"

template <typename T, typename Hasher>
void sort(GPUInfo const *const info, T *const data, u32 const length, Hasher hasher) {
  // u32(*counters)[DIGIT_PLACES][RADIX];
  Counters *counters;
  cudaMalloc(&counters, sizeof(*counters));
  cudaMemset(counters, 0, sizeof(*counters));

  u32 const blocks_per_grid = min(info->multiprocessors * counting::blocks_per_sm,
                                  (length + counting::slice_length - 1) / counting::slice_length);
  printf(
      "info:\n\tblocks per grid: %d\n\tthreads per block: %d\n\tmultiprocessors: %d\n\tcores per multiprocessor: %d\n",
      blocks_per_grid, counting::threads_per_block, info->multiprocessors, info->cores_per_multiprocessor);

  upfront_count<<<blocks_per_grid, counting::threads_per_block>>>(data, length, &counters->counters, hasher);
  handle_cuda_error();

  exclusive_scan<<<DIGIT_PLACES, RADIX>>>(counters, length);
  handle_cuda_error();

  decltype(counters->degenerates) degenerates;
  cudaMemcpy(degenerates, counters->degenerates, sizeof(degenerates), cudaMemcpyDeviceToHost);

  u32 *atomic_counter;
  cudaMalloc(&atomic_counter, sizeof(u32));
  cudaMemset(atomic_counter, 0, sizeof(u32));

  u32 const slices = (length + sorting::slice_length - 1) / sorting::slice_length;
  sorting::StatusCounter *status_counters;
  cudaMalloc(&status_counters, RADIX * slices * sizeof(sorting::StatusCounter));
  cudaMemset(status_counters, 0, RADIX * slices * sizeof(sorting::StatusCounter));

  // onesweep iterations
  for (u32 place = 0; place < ARRAY_LENGTH(counters->counters); ++place) {
    if (degenerates[place])
      continue;
    sorting::onesweep<<<slices, sorting::threads_per_block>>>(atomic_counter, data, length, place, status_counters,
                                                              &counters->counters[place], hasher);
  }

  for (u32 place = 0; place < ARRAY_LENGTH(degenerates); ++place) {
    printf("%d, ", degenerates[place]);
    // if (degenerates[place]
  }
  printf("\n");
  // debug copyback
  auto cpu_counters = new u32[1][DIGIT_PLACES][RADIX];
  cudaMemcpy(cpu_counters, counters->counters, sizeof(counters->counters), cudaMemcpyDeviceToHost);
  for (u32 place = 0; place < DIGIT_PLACES; ++place) {
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

  cudaFree(counters);
  cudaFree(atomic_counter);
  cudaFree(status_counters);
}
