#pragma once

#include "common.h"
#include <bit>
#include <concepts>
#include <cuda/std/bit>

#define gpu __device__
#define global __global__
#define shared __shared__

// size of the hash in bits
constexpr u32 KEY_SIZE = sizeof(u32) * 8;
// size of the radix in bits
constexpr u32 RADIX_SIZE = 8; // < KEY_SIZE
static_assert(KEY_SIZE > 0 && RADIX_SIZE > 0, "KEY_SIZE and RADIX_SIZE must be greater than 0");
static_assert(RADIX_SIZE < KEY_SIZE, "RADIX_SIZE must be less than KEY_SIZE");

// TODO: assert < maxint

// number of radix values
constexpr u32 RADIX = 1u << RADIX_SIZE;
static_assert(RADIX % 32 == 0, "RADIX must be a multiple of 32");
// digit places in a hash
constexpr u32 DIGIT_PLACES = KEY_SIZE / RADIX_SIZE;
// all 1 in the first RADIX_SIZE bits
constexpr u32 RADIX_MASK = RADIX - 1;

constexpr u32 SYNC_WARP = 0xffffffff;

namespace counting {
inline constexpr u32 blocks_per_sm = 2;
inline constexpr u32 threads_per_block = 256;
inline constexpr u32 items_per_thread_per_slice = 8;
inline constexpr u32 slice_length = threads_per_block * items_per_thread_per_slice;
} // namespace counting

namespace sorting {
// inline constexpr u32 blocks_per_sm = 2;
inline constexpr u32 threads_per_block = 256;
static_assert(threads_per_block % 32 == 0, "threads_per_block must be a multiple of 32");
static_assert(threads_per_block >= 32, "threads_per_block must be at least 32");

inline constexpr u32 items_per_thread = 8;

// not parameters not shared, move in onesweep
inline constexpr u32 slice_length = threads_per_block * items_per_thread;
inline constexpr u32 warp_slices = threads_per_block / 32;
static_assert(warp_slices <= 32, "warp_slices must be at most 32");

// (32 - count_leading_zeroes(n-1)) calculates the smallest power of two >=n
// undefined behaviour for clz(0)
inline constexpr u32 subwarp_size =
    (sorting::warp_slices == 1) ? 1U : 1 << (32 - std::countl_zero(sorting::warp_slices - 1));
} // namespace sorting

template <typename Func, typename T>
concept is_hasher = requires(Func f, T const &val) {
  // enforces that f(val) is valid and returns exactly an u32
  { f(val) } -> std::same_as<u32>;
};

void handle_cuda_error() {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

// intra-warp inclusive prefix sum using register shuffles
template <u32 width> gpu inline u32 warp_inclusive_multiscan(u32 val) {
  static_assert(cuda::std::has_single_bit(width), "Width of warp_inclusive_scan must be a power of two");
  static_assert(width <= 32, "Width of warp_inclusive_scan must <= 32");
#pragma unroll
  for (u32 offset = 1; offset < width; offset <<= 1) {
    u32 const remote_val = __shfl_up_sync(SYNC_WARP, val, offset, width);
    if ((threadIdx.x % width) >= offset) {
      val += remote_val;
    }
  }
  return val;
}
