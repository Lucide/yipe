#pragma once

#include <cuda/std/optional>
#include "scan/scan.cuh"

// size of the radix in bits
constexpr u8 RADIX_SIZE = 8;
static_assert(RADIX_SIZE > 0, "RADIX_SIZE must be greater than 0");

// number of radix values
constexpr u16 RADIX = 1u << RADIX_SIZE;
static_assert(RADIX % 32 == 0, "RADIX must be a multiple of 32");
static_assert(RADIX <= 1024, "RADIX can't be higher than 1024");

template <typename Func, typename T>
concept is_hasher = requires(Func f, T const val) {
  // enforces that f(val) is valid and returns an unsigned integer
  { f(val) } -> std::unsigned_integral;
};

template <typename Hasher, typename Data>
  requires is_hasher<Hasher, Data>
struct Key {
  using KeyType = std::invoke_result_t<Hasher, Data>;
  // size of the hash in bits
  static constexpr u8 key_size = sizeof(KeyType) * 8;
  static_assert(RADIX_SIZE < key_size, "RADIX_SIZE must be less than KEY_SIZE");
  // digit places in a hash
  static constexpr u8 digit_places = key_size / RADIX_SIZE;
  static constexpr u16 total_bits = std::numeric_limits<KeyType>::digits;

  // mask generation safe for bit counts up to total_bits
  GPU HOST static constexpr KeyType mask(u16 num_bits) {
    if (num_bits >= total_bits)
      return ~KeyType(0);
    return (KeyType(1) << num_bits) - KeyType(1);
  }

  // all 1 in the first RADIX_SIZE bits
  static constexpr KeyType radix_mask = mask(RADIX_SIZE);

  GPU HOST static constexpr u16 extract_digit(KeyType key, u8 place) {
    const auto shift = static_cast<KeyType>(place * RADIX_SIZE);
    return (key >> shift) & radix_mask;
  }
};

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

template <IsInstanceOf<Key> Key> struct Counters {
  u32 counters[Key::digit_places][RADIX];
  bool degenerates[Key::digit_places];
};

template <bool detect = false>
GPU inline std::conditional_t<detect, cuda::std::optional<u32>, u32>
radix_scan(u32 const digit_count, std::conditional_t<detect, u32, std::monostate> const length = std::monostate{}) {
  // threads must be within the radix range
  assert(threadIdx.x < RADIX);

  constexpr u8 warps = RADIX / 32;
  u8 const lane = threadIdx.x % 32;
  u8 const warp_id = static_cast<u8>(threadIdx.x / 32);

  // shared memory to store the aggregate sum of each warp
  SHARED u32 warp_counters[warps];                                      // NOLINT(bugprone-dynamic-static-initializers)
  SHARED std::conditional_t<detect, u32[warps], char> degenerate_warps; // NOLINT(bugprone-dynamic-static-initializers)

  // scan within each warp
  bool degenerate_warp;
  if constexpr (detect) {
    // detect if the warp is degenerate, i.e. only one digit population
    degenerate_warp = __any_sync(SYNC_WARP, digit_count == length);
  }
  u32 sum = warp_inclusive_multiscan(digit_count);

  // last thread in each warp publishes the warp total
  if (lane == 31) {
    if constexpr (detect) {
      degenerate_warps[warp_id] = degenerate_warp;
    }
    warp_counters[warp_id] = sum;
  }

  __syncthreads();

  SHARED bool degenerate_block; // NOLINT(bugprone-dynamic-static-initializers)

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
