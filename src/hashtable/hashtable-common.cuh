#pragma once

#include <cuda/std/optional>
#include "common.h"

struct Bucket {
  u32 begin, end;
};

struct HashTable {
  u32 length;
  u32 table[1];

  template <std::unsigned_integral Hash> GPU cuda::std::optional<Bucket> lookup(const Hash hash) const {
    assert(hash + 1 < this->length);
    const u32 begin = this->table[hash];
    const u32 end = this->table[hash + 1];

    if (begin == 0 || end == 0 || begin == end) {
      return cuda::std::nullopt;
    }

    return Bucket{begin - 1, end - 1};
  }
};

namespace hash_table {
inline constexpr u8 blocks_per_sm = 2;
inline constexpr u16 threads_per_block = 256;
inline constexpr u16 items_per_thread_per_slice = 8;
} // namespace hash_table
