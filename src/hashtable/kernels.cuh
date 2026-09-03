#pragma once

#include "hashtable/hashtable-common.cuh"
#include "sort/sort-common.cuh"

struct Bucket {
  u32 begin;
  u32 end;
};

struct HashTable {
  u32 length;
  u32 table[0];

  template <std::unsigned_integral Hash> GPU cuda::std::optional<Bucket> lookup(Hash hash) {
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
inline constexpr u32 slice_length = threads_per_block * items_per_thread_per_slice;

template <typename Data, typename Hasher>
  requires is_hasher<Hasher, Data>
KERNEL void build(Data *const data, u32 const length, HashTable *hash_table, Hasher hasher) {
  using Key = Key<Hasher, Data>;
  hash_table->length = length + 1;
  cuda::std::span<u32, cuda::std::dynamic_extent> table(hash_table->table, length + 1);
  assert(length > 0);

  for (u32 i = blockIdx.x * slice_length; i < length; i += gridDim.x * slice_length) {
    // iterate over the data slice in strides of block size
    // check the slice fits inside length-1, we are reading two points at a time
    u32 const slice_end = min(length - 1, i + slice_length);
    for (u32 j = i + threadIdx.x; j < slice_end; j += blockDim.x) {
      Data const item_left = data[j];
      typename Key::KeyType const hash_left = hasher(item_left);
      assert(hash_left < length); // invalid hash function
      Data const item_right = data[j + 1];
      typename Key::KeyType const hash_right = hasher(item_right);
      assert(hash_right < length);     // invalid hash function
      assert(hash_left <= hash_right); // unsorted data

      if (hash_left != hash_right) {
        // next hash after left
        const u32 left_next = hash_left + 1;
        // end of the left bucket (+1)
        const u32 left_end = j + 2;

        // publish end of left bucket
        table[left_next] = left_end;
        // if the right bucket *is* left_next, nothing to do
        if (hash_right > left_next) {
          // the start of the right bucket is the end of the left bucket
          table[hash_right] = left_end;
        }
      }
    }
  }
}
} // namespace hash_table
