#pragma once

#include "chained-scan.cuh"

template <std::integral Data> void scan(const Data *const src, Data *const dst, u32 const length) {
  u32 *atomic_counter;
  handle_cuda_result(cudaMallocAsync(&atomic_counter, sizeof(u32), cudaStreamPerThread), "Prefix: Allocate id counter");
  handle_cuda_error_debug("Prefix: Allocate id counter");
  handle_cuda_result(cudaMemsetAsync(atomic_counter, 0, sizeof(u32)), "Prefix: Memset id counter");
  handle_cuda_error_debug("Prefix: Memset id counter");
  u32 const slices = (length + chained_scan::slice_max_length - 1) / chained_scan::slice_max_length;
  u32 const states_size = slices * sizeof(chained_scan::State);
  u32 const counters_prefixes_size = 2 * slices * sizeof(Data);
  chained_scan::State *states;
  Data *counters_prefixes;
  handle_cuda_result(cudaMallocAsync(&states, states_size, cudaStreamPerThread), "Prefix: Allocate states");
  handle_cuda_error_debug("Prefix: Allocate states");
  handle_cuda_result(cudaMallocAsync(&counters_prefixes, counters_prefixes_size, cudaStreamPerThread),
                     "Prefix: Allocate counters");
  handle_cuda_error_debug("Prefix: Allocate counters");
  handle_cuda_result(cudaMemsetAsync(states, 0, states_size), "Prefix: Memset states");
  handle_cuda_error_debug("Prefix: Memset states");
  //   cudaMemset(counters, 0, counters_size);
  //   cudaMemset(prefixes, 0, counters_size);

  chained_scan::chained_scan<<<slices, chained_scan::threads_per_block>>>(atomic_counter, src, dst, length, states,
                                                                          counters_prefixes);
  handle_cuda_error_debug("Prefix: `chained_scan` kernel");

  handle_cuda_result(cudaFreeAsync(atomic_counter, cudaStreamPerThread), "Prefix: Free id counter");
  handle_cuda_error_debug("Prefix: Free id counter");
  handle_cuda_result(cudaFreeAsync(states, cudaStreamPerThread), "Prefix: Free states");
  handle_cuda_error_debug("Prefix: Free states");
  handle_cuda_result(cudaFreeAsync(counters_prefixes, cudaStreamPerThread), "Prefix: Free counters");
  handle_cuda_error_debug("Prefix: Free counters");
}
