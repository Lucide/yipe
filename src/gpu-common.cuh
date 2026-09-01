#pragma once

#include <cuda/std/bit>
#include <cuda/std/span>
#include "common.h"

template <typename T, template <typename...> class TargetTemplate> struct is_instance_of : std::false_type {};

template <template <typename...> class TargetTemplate, typename... Args>
struct is_instance_of<TargetTemplate<Args...>, TargetTemplate> : std::true_type {};

template <typename T, template <typename...> class TargetTemplate>
concept IsInstanceOf = is_instance_of<T, TargetTemplate>::value;

enum class Status : u32 {
  Invalid = 0,
  Count = 1,
  Prefix = 2,
};

struct StatusCounter {
  Status status : 2;
  u32 counter : 30;
};

static_assert(sizeof(StatusCounter) == sizeof(u32), "status and counter must be packed in a single 32 bit word");

constexpr u32 SYNC_WARP = 0xffffffff;

void handle_cuda_error() {
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

template <typename T, size_t N1, size_t N2> gpu constexpr auto flatten(T (&arr)[N1][N2]) {
  return cuda::std::span<T, N1 * N2>(&arr[0][0], N1 * N2);
}

// #define __CUDA_ARCH__ 700
// #define __CUDA_ARCH__ 610
gpu u32 match_any_polyfill(u32 thread_mask, u32 key) {
#if __CUDA_ARCH__ >= 700
  // sm_70+ volta, turing, ...
  return __match_any_sync(thread_mask, key);
#else
  u32 candidates = thread_mask;
  for (u32 mask = 1; mask < RADIX; mask <<= 1) {
    u32 bit = key & mask;
    candidates &= __ballot_sync(thread_mask, static_cast<int>(bit)) ^ (UINT_MAX * !bit);
  }
  return candidates;
#endif
}

gpu void nanosleep_polyfill(const unsigned int ns) {
  // in pascal there is no nanosleep
  // it is mostly needed on post-volta gpus
#if __CUDA_ARCH__ >= 700
  __nanosleep(ns);
#endif
}

gpu host constexpr u32 pow2_ceiling(const u32 val) {
  // (32 - count_leading_zeroes(n-1)) calculates the smallest power of two >=n
  // undefined behaviour for clz(0)
  return (val == 1) ? 1u : 1u << (32u - cuda::std::countl_zero(val - 1u));
}
