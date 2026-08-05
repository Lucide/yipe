#ifndef YIPE_GPU_COMMON_H
#define YIPE_GPU_COMMON_H

#include "common.h"
#include <concepts>

#define gpu __device__
#define global __global__
#define shared __shared__

// size of the hash in bits
u32 constexpr KEY_SIZE = sizeof(u32) * 8;
// size of the radix in bits
u32 constexpr RADIX_SIZE = 8; // < KEY_SIZE
static_assert(KEY_SIZE > 0 && RADIX_SIZE > 0, "KEY_SIZE and RADIX_SIZE must be greater than 0");
static_assert(RADIX_SIZE < KEY_SIZE, "RADIX_SIZE must be less than KEY_SIZE");

// TODO: assert < maxint

// number of radix values
u32 constexpr RADIX = 1u << RADIX_SIZE;
// digit places in a hash
u32 constexpr DIGIT_PLACES = KEY_SIZE / RADIX_SIZE;
// all 1 in the first RADIX_SIZE bits
u32 constexpr RADIX_MASK = RADIX - 1;

u32 constexpr SYNC_WARP = 0xffffffff;

u32 constexpr COUNTING_BLOCKS_PER_SM = 2;
u32 constexpr COUNTING_THREADS_PER_BLOCK = 256;
u32 constexpr COUNTING_ITEMS_PER_THREAD = 8;
u32 constexpr COUNTING_SLICE_SIZE = COUNTING_THREADS_PER_BLOCK * COUNTING_ITEMS_PER_THREAD;

u32 constexpr SORTING_BLOCKS_PER_SM = 2;
u32 constexpr SORTING_THREADS_PER_BLOCK = 256;
u32 constexpr SORTING_ITEMS_PER_THREAD = 8;
u32 constexpr SORTING_SLICE_SIZE = SORTING_THREADS_PER_BLOCK * SORTING_ITEMS_PER_THREAD;

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

#endif
