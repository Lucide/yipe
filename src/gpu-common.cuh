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

GPU HOST int3 point_discretize(Point const point, const float particle_diameter) {
  return {static_cast<int>(floorf(point.x / particle_diameter)), static_cast<int>(floorf(point.y / particle_diameter)),
          static_cast<int>(floorf(point.z / particle_diameter))};
}

GPU HOST u32 point_hash(Point const point, u32 const particle_count, const float particle_diameter) {
  static u32 const primes[3] = {73856093, 19349663, 83492791};
  int3 discrete = point_discretize(point, particle_diameter);
  return ((static_cast<u32>(discrete.x) * primes[0]) ^ (static_cast<u32>(discrete.y) * primes[1]) ^
          (static_cast<u32>(discrete.z) * primes[2])) %
         particle_count;
}

struct PointHasher {
  u32 length;
  float particle_diameter;

  GPU HOST u32 operator()(const Point p) const { return point_hash(p, length, particle_diameter); }
};

GPU HOST PointHasher get_hasher(const u32 length, const float particle_diameter) {
  return PointHasher{length, particle_diameter};
}

// #define __CUDA_ARCH__ 700
// #define __CUDA_ARCH__ 610
template <u8 size> GPU u32 match_any_polyfill(u16 const key, u32 const thread_mask = SYNC_WARP) {
#if __CUDA_ARCH__ >= 700
  // sm_70+ volta, turing, ...
  return __match_any_sync(thread_mask, key);
#else
  CPRINT("match_any polyfilled (__CUDA_ARCH__=" XSTR(__CUDA_ARCH__) ")")
  u32 candidates = thread_mask;
  for (u16 mask = 1; mask < 1 << size; mask <<= 1) {
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
  return (val == 1) ? 1u : static_cast<u8>(1u << (32 - cuda::std::countl_zero(val - 1u)));
}
