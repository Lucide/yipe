#pragma once

#include "chained-scan.cuh"

template <std::integral Data>
void scan(Data const *const src, Data *const dst, u32 const length, cudaStream_t const stream) {
  u32 *atomic_counter;
  TRAP_CUDAPI(cudaMallocAsync(&atomic_counter, sizeof(u32), stream), "prefix: allocate id counter");
  TRAP_CUDAPI(cudaMemsetAsync(atomic_counter, 0, sizeof(u32), stream), "prefix: memset id counter");
  u32 const slices = (length + chained_scan::slice_max_length - 1) / chained_scan::slice_max_length;
  u32 const states_size = slices * sizeof(chained_scan::State);
  u32 const counters_prefixes_size = 2 * slices * sizeof(Data);
  chained_scan::State *states;
  Data *counters_prefixes;
  TRAP_CUDAPI(cudaMallocAsync(&states, states_size, stream), "prefix: allocate states");
  TRAP_CUDAPI(cudaMallocAsync(&counters_prefixes, counters_prefixes_size, stream), "prefix: allocate counters");
  TRAP_CUDAPI(cudaMemsetAsync(states, 0, states_size, stream), "prefix: memset states");
  //   cudaMemset(counters, 0, counters_size);
  //   cudaMemset(prefixes, 0, counters_size);

  TRAP_KERNEL((chained_scan::chained_scan<<<slices, chained_scan::threads_per_block, 0, stream>>>(
                  atomic_counter, src, dst, length, states, counters_prefixes)),
              "prefix: `chained_scan` kernel");

  TRAP_CUDAPI(cudaFreeAsync(atomic_counter, stream), "prefix: free id counter");
  TRAP_CUDAPI(cudaFreeAsync(states, stream), "prefix: free states");
  TRAP_CUDAPI(cudaFreeAsync(counters_prefixes, stream), "prefix: free counters");
}
