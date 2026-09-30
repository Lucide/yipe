#pragma once

#include <driver_types.h>
#include "onesweep.cuh"
#include "upfront-pass.cuh"

template <typename Data, typename Hasher>
  requires is_hasher<Hasher, Data>
void sort(
    GPUInfo const *const info, Data *const data, u32 const length, Hasher const hasher, cudaStream_t const stream) {
  using Key = Key<Hasher, Data>;
  Counters<Key> *counters;

  handle_cuda_result(cudaMallocAsync(&counters, sizeof(*counters), stream), "Sort: Allocate global counters");
  handle_cuda_error_debug("Sort: Allocate global counters");
  handle_cuda_result(cudaMemsetAsync(counters, 0, sizeof(*counters), stream), "Sort: Memset global counters");
  handle_cuda_error_debug("Sort: Memset global counters");

  u32 const blocks_per_grid =
      min(info->multiprocessors * upfront::blocks_per_sm, ceil_div<u32, u16, u32>(length, upfront::slice_length));

  upfront::count<<<blocks_per_grid, upfront::threads_per_block, 0, stream>>>(data, length, counters->counters, hasher);
  handle_cuda_error_debug("Sort: `upfront_count` kernel");
  upfront::exclusive_scan<Key><<<Key::digit_places, RADIX, 0, stream>>>(counters, length);
  handle_cuda_error_debug("Sort: `exclusive_scan` kernel");

  u32 *atomic_counter;
  handle_cuda_result(cudaMallocAsync(&atomic_counter, sizeof(u32), stream), "Sort: Allocate id counter");
  handle_cuda_error_debug("Sort: Allocate id counter");
  handle_cuda_result(cudaMemsetAsync(atomic_counter, 0, sizeof(u32), stream), "Sort: Memset id counter");
  handle_cuda_error_debug("Sort: Memset id counter");
  u32 const slices = ceil_div<u32, u16, u32>(length, onesweep::slice_max_length);
  u32 const state_couters_size = RADIX * slices * sizeof(onesweep::StateCounter);
  onesweep::StateCounter *state_counters;
  handle_cuda_result(cudaMallocAsync(&state_counters, state_couters_size, stream), "Sort: Allocate state/counters");
  handle_cuda_error_debug("Sort: Allocate state/counters");
  Data *dst;
  handle_cuda_result(cudaMallocAsync(&dst, length * sizeof(Data), stream), "Sort: Allocate helper array");
  handle_cuda_error_debug("Sort: Allocate helper array");
  Data *src = data;

  decltype(counters->degenerates) degenerates;
  handle_cuda_result(
      cudaMemcpyAsync(degenerates, counters->degenerates, sizeof(degenerates), cudaMemcpyDeviceToHost, stream),
      "Sort: Download degenerate blocks");
  handle_cuda_result(cudaStreamSynchronize(stream), "Sort: Download degenerate blocks");
  // onesweep iterations
  for (u8 place = 0; place < Key::digit_places; ++place) {
    if (degenerates[place])
      continue;
    handle_cuda_result(cudaMemsetAsync(state_counters, 0, state_couters_size, stream), "Sort: Memset state/counters");
    onesweep::onesweep<<<slices, onesweep::threads_per_block, 0, stream>>>(
        atomic_counter, src, dst, length, place, state_counters, counters->counters[place], hasher);
    handle_cuda_error_debug("Sort: `onesweep` kernel");
    std::swap(src, dst);
  }

  if (dst == data) {
    handle_cuda_result(
        cudaMemcpyAsync((void *)data, (void *)src, length * sizeof(Data), cudaMemcpyDeviceToDevice, stream),
        "Sort: Memcpy sorted points");
    handle_cuda_error_debug("Sort: Memcpy sorted points");
    handle_cuda_result(cudaFreeAsync((void *)src, stream), "Sort: Free helper array");
    handle_cuda_error_debug("Sort: Free helper array");
  } else {
    handle_cuda_result(cudaFreeAsync((void *)dst, stream), "Sort: Free helper array");
    handle_cuda_error_debug("Sort: Free helper array");
  }
  handle_cuda_result(cudaFreeAsync(counters, stream), "Sort: Free counters");
  handle_cuda_error_debug("Sort: Free counters");
  handle_cuda_result(cudaFreeAsync(atomic_counter, stream), "Sort: Free id counter");
  handle_cuda_error_debug("Sort: Free id counter");
  handle_cuda_result(cudaFreeAsync(state_counters, stream), "Sort: Free state/counters");
  handle_cuda_error_debug("Sort: Free state/counters");
}
