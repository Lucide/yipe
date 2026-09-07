#pragma once

#include <cmath>
#include <cuda/std/bit>
#include <cuda/std/span>
#include "common.h"

template <typename T, template <typename...> class TargetTemplate> struct is_instance_of : std::false_type {};

template <template <typename...> class TargetTemplate, typename... Args>
struct is_instance_of<TargetTemplate<Args...>, TargetTemplate> : std::true_type {};

template <typename T, template <typename...> class TargetTemplate>
concept IsInstanceOf = is_instance_of<T, TargetTemplate>::value;

constexpr u32 SYNC_WARP = 0xffffffff;

template <std::integral T, std::integral U, std::integral V> GPU HOST constexpr V ceil_div(const T lhs, const U rhs) {
  const V h_lhs = static_cast<V>(lhs);
  const V h_rhs = static_cast<V>(rhs);
  return (h_lhs + (h_rhs - 1)) / h_rhs;
}

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

GPU HOST INLINE float sq_prod3(const Point lhs, const Point rhs) {
  return fmaf(lhs.x, rhs.x, fmaf(lhs.y, rhs.y, lhs.z * rhs.z));
}

GPU HOST INLINE constexpr mat4x4 transpose(const mat4x4 op) {
  return {
      .rows = {
               {op.comp.m00, op.comp.m10, op.comp.m20, op.comp.m30},
               {op.comp.m01, op.comp.m11, op.comp.m21, op.comp.m31},
               {op.comp.m02, op.comp.m12, op.comp.m22, op.comp.m32},
               {op.comp.m03, op.comp.m13, op.comp.m23, op.comp.m33},
               }
  };
}

GPU HOST INLINE constexpr Point operator+(const Point lhs, const Point rhs) {
  return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, 1};
}

GPU HOST INLINE constexpr Point operator-(const Point lhs, const Point rhs) {
  return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, 1};
}

GPU HOST INLINE constexpr Point operator*(const Point lhs, const float rhs) {
  return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, 1};
}
GPU HOST INLINE constexpr Point operator/(const Point lhs, const float rhs) {
  return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, 1};
}

GPU HOST INLINE float operator*(const float4 lhs, const float4 rhs) {
  return fmaf(lhs.x, rhs.x, fmaf(lhs.y, rhs.y, fmaf(lhs.z, rhs.z, lhs.w * rhs.w)));
}

GPU HOST INLINE constexpr bool operator<(const Point lhs, const Point rhs) {
  if (lhs.x < rhs.x)
    return true;
  else if (lhs.x > rhs.x)
    return false;

  if (lhs.y < rhs.y)
    return true;
  else if (lhs.y > rhs.y)
    return false;

  if (lhs.z < rhs.z)
    return true;
  else if (lhs.z > rhs.z)
    return false;

  return false;
}

GPU HOST INLINE float norm(const float4 v) { return sqrtf(v * v); }

GPU HOST INLINE float sq_dist(const Point lhs, const Point rhs) {
  const Point diff = lhs - rhs;
  return sq_prod3(diff, diff);
}

GPU HOST INLINE float dist(const float4 lhs, const float4 rhs) { return sqrt(sq_dist(lhs, rhs)); }

GPU HOST INLINE float4 operator*(const mat4x4 lhs, const float4 rhs) {
  return {lhs.rows.r0 * rhs, lhs.rows.r1 * rhs, lhs.rows.r2 * rhs, lhs.rows.r3 * rhs};
}

GPU HOST INLINE mat4x4 operator*(const mat4x4 lhs, const mat4x4 rhs) {
  mat4x4 rhs_t = transpose(rhs);
  return {
      .rows = {{lhs.rows.r0 * rhs_t.rows.r0, lhs.rows.r0 * rhs_t.rows.r1, lhs.rows.r0 * rhs_t.rows.r2,
                lhs.rows.r0 * rhs_t.rows.r3},
               {lhs.rows.r1 * rhs_t.rows.r0, lhs.rows.r1 * rhs_t.rows.r1, lhs.rows.r1 * rhs_t.rows.r2,
                lhs.rows.r1 * rhs_t.rows.r3},
               {lhs.rows.r2 * rhs_t.rows.r0, lhs.rows.r2 * rhs_t.rows.r1, lhs.rows.r2 * rhs_t.rows.r2,
                lhs.rows.r2 * rhs_t.rows.r3},
               {lhs.rows.r3 * rhs_t.rows.r0, lhs.rows.r3 * rhs_t.rows.r1, lhs.rows.r3 * rhs_t.rows.r2,
                lhs.rows.r3 * rhs_t.rows.r3}}
  };
}

GPU HOST constexpr float3 project4to3(const Point point) { return {point.x, point.y, point.z}; }
