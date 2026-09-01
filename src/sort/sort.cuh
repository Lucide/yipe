#pragma once

#include <cstdio>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "onesweep.cuh"
#include "upfront-pass.cuh"

template <typename Data, typename Hasher>
// requires is_hasher<Hasher, Data>
void sort(GPUInfo const *const info, Data *const data, u32 const length, Hasher hasher) {
  using Key = Key<Hasher, Data>;
  Counters<Key> *counters;

  cudaMalloc(&counters, sizeof(*counters));
  cudaMemset(counters, 0, sizeof(*counters));

  u32 const blocks_per_grid =
      min(info->multiprocessors * upfront::blocks_per_sm, (length + upfront::slice_length - 1) / upfront::slice_length);

  upfront::count<<<blocks_per_grid, upfront::threads_per_block>>>(data, length, counters->counters, hasher);
  handle_cuda_error();
  upfront::exclusive_scan<Key><<<Key::digit_places, RADIX>>>(counters, length);
  handle_cuda_error();

  decltype(counters->degenerates) degenerates;
  cudaMemcpy(degenerates, counters->degenerates, sizeof(degenerates), cudaMemcpyDeviceToHost);

  u32 *atomic_counter;
  cudaMalloc(&atomic_counter, sizeof(u32));
  cudaMemset(atomic_counter, 0, sizeof(u32));
  u32 const slices = (length + onesweep::slice_length - 1) / onesweep::slice_length;
  u32 const status_couters_size = RADIX * slices * sizeof(StatusCounter);
  StatusCounter *status_counters;
  cudaMalloc(&status_counters, status_couters_size);
  Data *dst;
  if (auto err = cudaMalloc(&dst, sizeof(Data) * length); err != cudaSuccess)
    printf("alloc failed: %s\n", cudaGetErrorString(err));
  auto src = data;

  // await pending device operations
  cudaDeviceSynchronize();

  // onesweep iterations
  for (u8 place = 0; place < ARRAY_LENGTH(counters->counters); ++place) {
    if (degenerates[place])
      continue;
    printf("place: %d\n", place);
    cudaMemset(status_counters, 0, status_couters_size);
    onesweep::onesweep<<<slices, onesweep::threads_per_block>>>(atomic_counter, src, dst, length, place,
                                                                status_counters, counters->counters[place], hasher);
    handle_cuda_error();
    std::swap(src, dst);
  }

  if (dst == data) {
    cudaMemcpy(data, src, sizeof(Data) * length, cudaMemcpyDeviceToDevice);
    cudaFree(src);
  } else {
    cudaFree(dst);
  }

  // for (u32 place = 0; place < ARRAY_LENGTH(degenerates); ++place) {
  //   printf("%d, ", degenerates[place]);
  //   // if (degenerates[place]
  // }
  // printf("\n");
  // // debug copyback
  // auto cpu_counters = new u32[1][DIGIT_PLACES][RADIX];
  // cudaMemcpy(cpu_counters, counters->counters, sizeof(counters->counters), cudaMemcpyDeviceToHost);
  // for (u32 place = 0; place < DIGIT_PLACES; ++place) {
  //   if (degenerates[place])
  //     continue;
  //   printf("place %d: [", place);
  //   for (u32 digit = 0; digit < RADIX; ++digit) {
  //     printf("%d:%d,", digit, (*cpu_counters)[place][digit]);
  //   }
  //   printf("]\n");
  // }
  // delete[] cpu_counters;
  // // end debug copyback

  cudaFree(counters);
  cudaFree(atomic_counter);
  cudaFree(status_counters);
}
