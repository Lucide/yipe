#pragma once

#include <cuda/std/bit>
#include <cuda/std/span>
#include "common.h"

template <typename T, template <typename...> class TargetTemplate> struct is_instance_of : std::false_type {};

template <template <typename...> class TargetTemplate, typename... Args>
struct is_instance_of<TargetTemplate<Args...>, TargetTemplate> : std::true_type {};

template <typename T, template <typename...> class TargetTemplate>
concept IsInstanceOf = is_instance_of<T, TargetTemplate>::value;

constexpr u32 SYNC_WARP = 0xffffffff;

template <typename T, size_t N1, size_t N2> GPU constexpr auto flatten(T (&arr)[N1][N2]) {
  return cuda::std::span<T, N1 * N2>(&arr[0][0], N1 * N2);
}

// #define __CUDA_ARCH__ 700
// #define __CUDA_ARCH__ 610
template <u8 width> GPU u32 match_any_polyfill(u32 thread_mask, u16 key) {
#if __CUDA_ARCH__ >= 700
  // sm_70+ volta, turing, ...
  return __match_any_sync(thread_mask, key);
#else
  CPRINT("match_any polyfilled (__CUDA_ARCH__=" XSTR(__CUDA_ARCH__) ")")
  u32 candidates = thread_mask;
  for (u16 mask = 1; mask < 1 << width; mask <<= 1) {
    u16 bit = key & mask;
    candidates &= __ballot_sync(thread_mask, static_cast<int>(bit)) ^ (UINT_MAX * !bit);
  }
  return candidates;
#endif
}

GPU void nanosleep_polyfill(const unsigned int ns) {
// in pascal there is no nanosleep
// it is mostly needed on post-volta gpus
#if __CUDA_ARCH__ >= 700
  __nanosleep(ns);
#else
  CPRINT("nanosleep polyfilled (__CUDA_ARCH__=" XSTR(__CUDA_ARCH__) ")")
#endif
}

GPU HOST constexpr u8 pow2_ceiling(const u8 val) {
  // (32 - count_leading_zeroes(n-1)) calculates the smallest power of two >=n
  // undefined behaviour for clz(0)
  return (val == 1) ? 1u : 1u << (32u - cuda::std::countl_zero(val - 1u));
}
