#pragma once

#include "hashtable/hashtable-common.cuh"
#include "sort/sort-common.cuh"

namespace hash_table {
inline constexpr u32 slice_length = threads_per_block * items_per_thread_per_slice;

template <typename Data, typename Hasher>
  requires is_hasher<Hasher, Data>
KERNEL void
build(const Data *const data, const u32 length, HashTable *const hash_table, const u32 buckets, const Hasher hasher) {
  using Key = Key<Hasher, Data>;
  hash_table->length = buckets + 1;
  cuda::std::span<u32, cuda::std::dynamic_extent> table(hash_table->table, buckets + 1);
  assert(length > 0);

  for (u32 i = blockIdx.x * slice_length; i < length; i += gridDim.x * slice_length) {
    // iterate over the data slice in strides of block size
    // if first element[0], add manually.
    u32 loop_start = i + threadIdx.x;
    if (loop_start == 0) {
      const Data item = data[0];
      typename Key::KeyType const hash = hasher(item);
      table[hash] = 1;
    }
    // check the slice fits inside length-1, we are reading two points at a time
    u32 const slice_end = min(length - 1, i + slice_length);
    u32 j;
    for (j = loop_start; j < slice_end; j += blockDim.x) {
      Data const item_left = data[j];
      typename Key::KeyType const hash_left = hasher(item_left);
      assert(hash_left < buckets); // invalid hash function
      Data const item_right = data[j + 1];
      typename Key::KeyType const hash_right = hasher(item_right);
      assert(hash_right < buckets);    // invalid hash function
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
    if (j == length - 1) {
      const Data item = data[j];
      typename Key::KeyType const hash = hasher(item);
      table[hash + 1] = length + 1;
    }
  }
}

template <typename Data, typename Hasher>
  requires is_hasher<Hasher, Data>
KERNEL void verify(const Data *const data, const u32 length, const HashTable *const hash_table, const Hasher hasher) {
  using Key = Key<Hasher, Data>;
  const u32 buckets = hash_table->length - 1;
  cuda::std::span<const u32, cuda::std::dynamic_extent> table(hash_table->table, buckets + 1);
  const u32 warps = blockDim.x / 32;
  const u32 warp = threadIdx.x / 32;
  const u32 lane = threadIdx.x % 32;

  for (u32 i = blockIdx.x * slice_length; i < length; i += gridDim.x * slice_length) {
    // iterate over the data slice in strides of block size
    // check the slice fits inside length-1, we are reading two points at a time
    u32 const slice_end = min(length, i + slice_length);
    for (u32 j = i + threadIdx.x; j < slice_end; j += blockDim.x) {
      const Data item = data[j];
      typename Key::KeyType const hash = hasher(item);
      const auto bucket = hash_table->lookup(hash);
      assert(bucket);
      assert(bucket->begin <= j);
      assert(j < bucket->end);
    }
  }

  for (u32 i = blockIdx.x * slice_length; i < buckets; i += gridDim.x * slice_length) {
    // iterate over the data slice in strides of block size
    // check the slice fits inside length-1, we are reading two points at a time
    u32 const slice_end = min(buckets, i + slice_length);
    for (u32 j = i + warp; j < slice_end; j += warps) {
      const auto bucket = hash_table->lookup(j);

      if (bucket) {
        for (u32 k = bucket->begin + lane; k < bucket->end; k += 32) {
          const Data item = data[k];
          typename Key::KeyType const hash = hasher(item);
          assert(j == hash);
        }
      }
    }
  }
}
} // namespace hash_table
