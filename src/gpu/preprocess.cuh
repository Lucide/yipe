#pragma once

#include "gpu-common.cuh"

namespace preprocess {
inline constexpr u8 blocks_per_sm = 2;
inline constexpr u16 threads_per_block = 256;
inline constexpr u16 items_per_thread_per_slice = 8;

inline constexpr u32 slice_length = threads_per_block * items_per_thread_per_slice;

template <typename F, typename T_IN, typename T_OUT>
concept Arrow = requires(F f, T_IN in) {
  { f(in) } -> std::same_as<T_OUT>;
};

template <typename T_IN, typename T_OUT, Arrow<T_IN, T_OUT> F>
KERNEL void map(T_IN const *const in, T_OUT *const out, u32 const length, F f) {
  // iterate over slices in strides of grid size
  for (u32 i = blockIdx.x * slice_length; i < length; i += gridDim.x * slice_length) {
    // iterate over the data slice in strides of block size
    u32 const slice_end = min(length, i + slice_length);
    for (u32 j = i + threadIdx.x; j < slice_end; j += blockDim.x) {
      out[j] = f(in[j]);
    }
  }
}

template <typename T_IN, typename T_OUT, Arrow<T_IN, T_OUT> F>
void preprocess(
    GPUInfo const &info, T_IN const *const in, T_OUT *const out, u32 const length, F f, cudaStream_t const stream) {
  u32 const blocks_per_grid = min(info.multiprocessors * blocks_per_sm, ceil_div<u32, u16, u32>(length, slice_length));
  TRAP_KERNEL((map<<<blocks_per_grid, threads_per_block, 0, stream>>>(in, out, length, f)),
              "preprocess: map float3 to float4");
}
} // namespace preprocess
