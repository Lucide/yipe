#pragma once

#include "chained-scan.cuh"

template <std::integral Data> void scan(Data *const data, u32 const length) {
  u32 *atomic_counter;
  handle_cuda_result(cudaMalloc(&atomic_counter, sizeof(u32)), "Prefix: Allocate id counter");
  handle_cuda_result(cudaMemset(atomic_counter, 0, sizeof(u32)), "Prefix: Memset id counter");
  u32 const slices = (length + chained_scan::slice_max_length - 1) / chained_scan::slice_max_length;
  u32 const states_size = slices * sizeof(chained_scan::State);
  u32 const counters_prefixes_size = 2 * slices * sizeof(Data);
  chained_scan::State *states;
  Data *counters_prefixes;
  handle_cuda_result(cudaMalloc(&states, states_size), "Prefix: Allocate states");
  handle_cuda_result(cudaMalloc(&counters_prefixes, counters_prefixes_size), "Prefix: Allocate counters");
  handle_cuda_result(cudaMemset(states, 0, states_size), "Prefix: Memset states");
  //   cudaMemset(counters, 0, counters_size);
  //   cudaMemset(prefixes, 0, counters_size);

  chained_scan::chained_scan<<<slices, chained_scan::threads_per_block>>>(atomic_counter, data, length, states,
                                                                          counters_prefixes);
  handle_cuda_error("Prefix: `chained_scan` kernel");

  handle_cuda_result(cudaFree(atomic_counter), "Prefix: Free id counter");
  handle_cuda_result(cudaFree(states), "Prefix: Free states");
  handle_cuda_result(cudaFree(counters_prefixes), "Prefix: Free counters");
}
