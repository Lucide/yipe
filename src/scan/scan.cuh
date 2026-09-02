#pragma once

#include "chained-scan.cuh"

template <std::integral Data> void scan(GPUInfo const *const info, Data *const data, u32 const length) {
  u32 *atomic_counter;
  cudaMalloc(&atomic_counter, sizeof(u32));
  cudaMemset(atomic_counter, 0, sizeof(u32));
  u32 const slices = (length + chained_scan::slice_max_length - 1) / chained_scan::slice_max_length;
  u32 const states_size = slices * sizeof(chained_scan::State);
  u32 const counters_prefixes_size = 2 * slices * sizeof(Data);
  chained_scan::State *states;
  Data *counters_prefixes;
  cudaMalloc(&states, states_size);
  cudaMalloc(&counters_prefixes, counters_prefixes_size);
  cudaMemset(states, 0, states_size);
  //   cudaMemset(counters, 0, counters_size);
  //   cudaMemset(prefixes, 0, counters_size);

  chained_scan::chained_scan<<<slices, chained_scan::threads_per_block>>>(atomic_counter, data, length, states,
                                                                          counters_prefixes);
  handle_cuda_error();

  cudaFree(states);
  cudaFree(counters_prefixes);
}
