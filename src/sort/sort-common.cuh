#pragma once

#include <cassert>
#include <concepts>
#include <cuda/std/bit>
#include <cuda/std/optional>
#include <cuda/std/span>
#include <variant>
#include "common.h"

// size of the hash in bits
constexpr u8 KEY_SIZE = sizeof(u32) * 8;
// size of the radix in bits
constexpr u8 RADIX_SIZE = 8;
static_assert(KEY_SIZE > 0 && RADIX_SIZE > 0, "KEY_SIZE and RADIX_SIZE must be greater than 0");
static_assert(RADIX_SIZE < KEY_SIZE, "RADIX_SIZE must be less than KEY_SIZE");

// number of radix values
constexpr u16 RADIX = 1u << RADIX_SIZE;
static_assert(RADIX % 32 == 0, "RADIX must be a multiple of 32");
static_assert(RADIX <= 1024, "RADIX can't be higher than 1024");
// digit places in a hash
constexpr u8 DIGIT_PLACES = KEY_SIZE / RADIX_SIZE;
// all 1 in the first RADIX_SIZE bits
constexpr u32 RADIX_MASK = RADIX - 1;

constexpr u32 SYNC_WARP = 0xffffffff;

namespace upfront {
inline constexpr u8 blocks_per_sm = 2;
inline constexpr u16 threads_per_block = 256;
inline constexpr u16 items_per_thread_per_slice = 8;
} // namespace upfront

namespace onesweep {
inline constexpr u16 threads_per_block = 256;
static_assert(threads_per_block % 32 == 0, "threads_per_block must be a multiple of 32");
static_assert(threads_per_block >= RADIX, "threads_per_block must be at least RADIX");

inline constexpr u16 items_per_thread = 8;
} // namespace onesweep

struct Counters {
  u32 counters[DIGIT_PLACES][RADIX];
  bool degenerates[DIGIT_PLACES];
};

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

template <typename T, size_t N1, size_t N2> __device__ constexpr auto flatten(T (&arr)[N1][N2]) {
  return cuda::std::span<T, N1 * N2>(&arr[0][0], N1 * N2);
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

template <bool detect = false>
gpu inline std::conditional_t<detect, cuda::std::optional<u32>, u32>
radix_scan(u32 const digit_count, std::conditional_t<detect, u32, std::monostate> const length = std::monostate{}) {
  // threads must be within the radix range
  assert(threadIdx.x < RADIX);

  constexpr u8 warps = RADIX / 32;
  u8 const lane = threadIdx.x % 32;
  u8 const warp_id = threadIdx.x / 32;

  // shared memory to store the aggregate sum of each warp
  shared u32 warp_counters[warps];                                      // NOLINT(bugprone-dynamic-static-initializers)
  shared std::conditional_t<detect, u32[warps], char> degenerate_warps; // NOLINT(bugprone-dynamic-static-initializers)

  // scan within each warp
  bool degenerate_warp;
  if constexpr (detect) {
    // detect if the warp is degenerate, i.e. only one digit population
    degenerate_warp = __any_sync(SYNC_WARP, digit_count == length);
  }
  u32 sum = warp_inclusive_multiscan<32>(digit_count);

  // last thread in each warp publishes the warp total
  if (lane == 31) {
    if constexpr (detect) {
      degenerate_warps[warp_id] = degenerate_warp;
    }
    warp_counters[warp_id] = sum;
  }

  __syncthreads();

  shared bool degenerate_block; // NOLINT(bugprone-dynamic-static-initializers)

  // warp 0 scans the warp totals
  if (warp_id == 0) {
    bool flag = false;
    if constexpr (detect) {
      flag = __any_sync(SYNC_WARP, (lane < warps) ? degenerate_warps[lane] : 0);
      if (lane == 0) {
        degenerate_block = flag;
      }
    }
    if (!flag) {
      u32 const sum = warp_inclusive_multiscan<warps>((lane < warps) ? warp_counters[lane] : 0);
      if (lane < warps) {
        warp_counters[lane] = sum;
      }
    }
  }

  __syncthreads();

  if constexpr (detect) {
    if (degenerate_block) {
      return cuda::std::nullopt;
    }
  }

  sum = __shfl_up_sync(SYNC_WARP, sum, 1);
  if (lane == 0) {
    sum = 0;
  }

  // add prefix sum of previous warps to local warp results
  if (warp_id > 0) {
    sum += warp_counters[warp_id - 1];
  }

  return sum;
}
