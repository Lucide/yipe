#pragma once

#include <cmath>
#include <cuda/std/bit>
#include <cuda/std/span>
#include "common.h"

constexpr u32 SYNC_WARP = 0xffffffff;

template <typename T, template <typename...> class TargetTemplate> struct is_instance_of : std::false_type {};

template <template <typename...> class TargetTemplate, typename... Args>
struct is_instance_of<TargetTemplate<Args...>, TargetTemplate> : std::true_type {};

template <typename T, template <typename...> class TargetTemplate>
concept IsInstanceOf = is_instance_of<T, TargetTemplate>::value;

template <std::integral T> GPU HOST constexpr auto utos(T value) noexcept {
  return static_cast<std::make_signed_t<T>>(value);
}

template <typename T = double> GPU HOST constexpr float dtof(double value) noexcept {
  return static_cast<float>(value);
}

template <std::integral T> GPU HOST constexpr float itof(T value) noexcept {
  static_assert(sizeof(T) <= 4, "itof restricted up to 4B integer");
  return static_cast<float>(value);
}

template <std::integral T, std::integral U, std::integral V>
GPU HOST constexpr V ceil_div(T const lhs, U const rhs) noexcept {
  V const h_lhs = static_cast<V>(lhs);
  V const h_rhs = static_cast<V>(rhs);
  return (h_lhs + (h_rhs - 1)) / h_rhs;
}

template <typename T, size_t N1, size_t N2> GPU constexpr auto flatten(T (&arr)[N1][N2]) noexcept {
  return cuda::std::span<T, N1 * N2>(&arr[0][0], N1 * N2);
}

GPU HOST int3 point_discretize(Point const point, float const particle_diameter) {
  return {static_cast<int>(floorf(point.x / particle_diameter)), static_cast<int>(floorf(point.y / particle_diameter)),
          static_cast<int>(floorf(point.z / particle_diameter))};
}

GPU HOST u32 point_hash(Point const point, u32 const particle_count, float const particle_diameter) {
  static u32 const primes[3] = {73856093, 19349663, 83492791};
  int3 discrete = point_discretize(point, particle_diameter);
  return ((static_cast<u32>(discrete.x) * primes[0]) ^ (static_cast<u32>(discrete.y) * primes[1]) ^
          (static_cast<u32>(discrete.z) * primes[2])) %
         particle_count;
}

struct PointHasher {
  u32 length;
  float particle_diameter;

  GPU HOST u32 operator()(Point const p) const { return point_hash(p, length, particle_diameter); }
};

GPU HOST PointHasher get_hasher(u32 const length, float const particle_diameter) {
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

GPU void nanosleep_polyfill(uint const ns) {
// in pascal there is no nanosleep
// it is mostly needed on post-volta gpus
#if __CUDA_ARCH__ >= 700
  __nanosleep(ns);
#else
  CPRINT("nanosleep polyfilled (__CUDA_ARCH__=" XSTR(__CUDA_ARCH__) ")")
#endif
}

GPU HOST constexpr u8 pow2_ceiling(u8 const val) noexcept {
  // (32 - count_leading_zeroes(n-1)) calculates the smallest power of two >=n
  // undefined behaviour for clz(0)
  return (val == 1) ? 1u : static_cast<u8>(1u << (32 - cuda::std::countl_zero(val - 1u)));
}

GPU HOST inline float sq_prod3(Point const lhs, Point const rhs) {
  return fmaf(lhs.x, rhs.x, fmaf(lhs.y, rhs.y, lhs.z * rhs.z));
}

GPU HOST inline constexpr mat4x4 transpose(mat4x4 const op) noexcept {
  return {
      .rows = {
               {op.comp.m00, op.comp.m10, op.comp.m20, op.comp.m30},
               {op.comp.m01, op.comp.m11, op.comp.m21, op.comp.m31},
               {op.comp.m02, op.comp.m12, op.comp.m22, op.comp.m32},
               {op.comp.m03, op.comp.m13, op.comp.m23, op.comp.m33},
               }
  };
}

GPU HOST inline constexpr Point operator+(Point const lhs, Point const rhs) noexcept {
  return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, 1};
}

GPU HOST inline constexpr Point operator-(Point const lhs, Point const rhs) noexcept {
  return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, 1};
}

GPU HOST inline constexpr Point operator*(Point const lhs, float const rhs) noexcept {
  return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, 1};
}
GPU HOST inline constexpr Point operator/(Point const lhs, float const rhs) noexcept {
  return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, 1};
}

GPU HOST inline float operator*(float4 const lhs, float4 const rhs) {
  return fmaf(lhs.x, rhs.x, fmaf(lhs.y, rhs.y, fmaf(lhs.z, rhs.z, lhs.w * rhs.w)));
}

GPU HOST inline constexpr bool operator<(Point const lhs, Point const rhs) noexcept {
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

GPU HOST inline float norm(float4 const v) { return sqrtf(v * v); }

GPU HOST inline float sq_dist(Point const lhs, Point const rhs) {
  Point const diff = lhs - rhs;
  return sq_prod3(diff, diff);
}

GPU HOST inline float dist(float4 const lhs, float4 const rhs) { return sqrt(sq_dist(lhs, rhs)); }

GPU HOST inline float4 operator*(mat4x4 const lhs, float4 const rhs) {
  return {lhs.rows.r0 * rhs, lhs.rows.r1 * rhs, lhs.rows.r2 * rhs, lhs.rows.r3 * rhs};
}

GPU HOST inline mat4x4 operator*(mat4x4 const lhs, mat4x4 const rhs) {
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

GPU HOST constexpr float3 project4to3(Point const point) noexcept { return {point.x, point.y, point.z}; }
