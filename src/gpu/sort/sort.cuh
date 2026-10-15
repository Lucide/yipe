#pragma once

#include <driver_types.h>
#include "onesweep.cuh"
#include "upfront-pass.cuh"

template <typename Data, is_hasher<Data> Hasher>
void sort(GPUInfo const &info, Data *const data, u32 const length, Hasher const hasher, cudaStream_t const stream) {
  using Key = Key<Data, Hasher>;
  Counters<Key> *counters;

  TRAP_CUDAPI(cudaMallocAsync(&counters, sizeof(*counters), stream), "sort: allocate global counters");
  TRAP_CUDAPI(cudaMemsetAsync(counters, 0, sizeof(*counters), stream), "sort: memset global counters");

  u32 const blocks_per_grid =
      min(info.multiprocessors * upfront::blocks_per_sm, ceil_div<u32, u16, u32>(length, upfront::slice_length));

  TRAP_KERNEL((upfront::count<<<blocks_per_grid, upfront::threads_per_block, 0, stream>>>(data, length,
                                                                                          counters->counters, hasher)),
              "sort: `upfront_count` kernel");
  TRAP_KERNEL((upfront::exclusive_scan<Key><<<Key::digit_places, RADIX, 0, stream>>>(counters, length)),
              "sort: `exclusive_scan` kernel");

  u32 *atomic_counter;
  TRAP_CUDAPI(cudaMallocAsync(&atomic_counter, sizeof(u32), stream), "sort: allocate id counter");
  TRAP_CUDAPI(cudaMemsetAsync(atomic_counter, 0, sizeof(u32), stream), "sort: memset id counter");
  u32 const slices = ceil_div<u32, u16, u32>(length, onesweep::slice_max_length);
  u32 const state_couters_size = RADIX * slices * sizeof(onesweep::StateCounter);
  onesweep::StateCounter *state_counters;
  TRAP_CUDAPI(cudaMallocAsync(&state_counters, state_couters_size, stream), "sort: allocate state/counters");
  Data *dst;
  TRAP_CUDAPI(cudaMallocAsync(&dst, ARRAY_SIZE(dst, length), stream), "sort: allocate helper array");
  Data *src = data;

  decltype(counters->degenerates) degenerates;
  TRAP_CUDAPI(cudaMemcpy(degenerates, counters->degenerates, sizeof(degenerates), cudaMemcpyDeviceToHost),
              "sort: download degenerate blocks");
  // onesweep iterations
  for (u8 place = 0; place < Key::digit_places; ++place) {
    if (degenerates[place])
      continue;
    TRAP_CUDAPI(cudaMemsetAsync(state_counters, 0, state_couters_size, stream), "sort: memset state/counters");
    TRAP_KERNEL((onesweep::onesweep<<<slices, onesweep::threads_per_block, 0, stream>>>(
                    atomic_counter, src, dst, length, place, state_counters, counters->counters[place], hasher)),
                "sort: `onesweep` kernel");
    cuda::std::swap(src, dst);
  }

  if (dst == data) {
    TRAP_CUDAPI(cudaMemcpyAsync((void *)data, (void *)src, ARRAY_SIZE(src, length), cudaMemcpyDeviceToDevice, stream),
                "sort: memcpy sorted points");
    TRAP_CUDAPI(cudaFreeAsync((void *)src, stream), "sort: free helper array");
  } else {
    TRAP_CUDAPI(cudaFreeAsync((void *)dst, stream), "sort: free helper array");
  }
  TRAP_CUDAPI(cudaFreeAsync(counters, stream), "sort: free counters");
  TRAP_CUDAPI(cudaFreeAsync(atomic_counter, stream), "sort: free id counter");
  TRAP_CUDAPI(cudaFreeAsync(state_counters, stream), "sort: free state/counters");
}
