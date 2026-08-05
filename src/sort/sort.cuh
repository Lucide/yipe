#ifndef YIPE_SORT_CUH
#define YIPE_SORT_CUH

#include "common.h"
#include "sort/upfront-pass.cuh"
#include <cstdio>

template <typename T, typename Hasher>
void sort(GPUInfo const *const info, T *const data, u32 const size, Hasher hasher) {
  u32(*counters)[DIGIT_PLACES][RADIX];
  cudaMalloc(&counters, sizeof(u32[DIGIT_PLACES][RADIX]));
  cudaMemset(counters, 0, sizeof(u32[DIGIT_PLACES][RADIX]));

  u32 const blocks_per_grid =
      min(info->multiprocessors * COUNTING_BLOCKS_PER_SM, (size + COUNTING_SLICE_SIZE - 1) / COUNTING_SLICE_SIZE);
  printf(
      "info:\n\tblocks per grid: %d\n\tthreads per block: %d\n\tmultiprocessors: %d\n\tcores per multiprocessor: %d\n",
      blocks_per_grid, COUNTING_THREADS_PER_BLOCK, info->multiprocessors, info->cores_per_multiprocessor);

  upfront_count<<<blocks_per_grid, COUNTING_THREADS_PER_BLOCK>>>(data, size, counters, hasher);
  handle_cuda_error();
  exclusive_scan<<<DIGIT_PLACES, RADIX>>>(counters);
  handle_cuda_error();

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
}

#endif // YIPE_SORT_CUH
