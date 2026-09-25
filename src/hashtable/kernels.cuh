#pragma once

#include "hashtable/hashtable-common.cuh"
#include "sort/sort-common.cuh"

namespace hash_table {
inline constexpr u32 slice_length = threads_per_block * items_per_thread_per_slice;

template <typename Data, typename Hasher>
  requires is_hasher<Hasher, Data>
KERNEL void
build(Data const *const data, u32 const length, HashTable *const hash_table, u32 const buckets, Hasher const hasher) {
  using Key = Key<Hasher, Data>;
  hash_table->length = buckets + 1;
  cuda::std::span<u32, cuda::std::dynamic_extent> table(hash_table->table, buckets + 1);
  assert(length > 0);

  for (u32 i = blockIdx.x * slice_length; i < length; i += gridDim.x * slice_length) {
    // iterate over the data slice in strides of block size
    // if first element[0], add manually.
    u32 loop_start = i + threadIdx.x;
    if (loop_start == 0) {
      Data const item = data[0];
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
        u32 const left_next = hash_left + 1;
        // end of the left bucket (+1)
        u32 const left_end = j + 2;

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
      Data const item = data[j];
      typename Key::KeyType const hash = hasher(item);
      table[hash + 1] = length + 1;
    }
  }
}

template <typename Data, typename Hasher>
  requires is_hasher<Hasher, Data>
KERNEL void verify(Data const *const data, u32 const length, HashTable const *const hash_table, Hasher const hasher) {
  using Key = Key<Hasher, Data>;
  u32 const buckets = hash_table->length - 1;
  cuda::std::span<u32 const, cuda::std::dynamic_extent> table(hash_table->table, buckets + 1);
  u32 const warps = blockDim.x / 32;
  u32 const warp = threadIdx.x / 32;
  u32 const lane = threadIdx.x % 32;

  for (u32 i = blockIdx.x * slice_length; i < length; i += gridDim.x * slice_length) {
    // iterate over the data slice in strides of block size
    // check the slice fits inside length-1, we are reading two points at a time
    u32 const slice_end = min(length, i + slice_length);
    for (u32 j = i + threadIdx.x; j < slice_end; j += blockDim.x) {
      Data const item = data[j];
      typename Key::KeyType const hash = hasher(item);
      auto const bucket = hash_table->lookup(hash);
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
      auto const bucket = hash_table->lookup(j);

      if (bucket) {
        for (u32 k = bucket->begin + lane; k < bucket->end; k += 32) {
          Data const item = data[k];
          typename Key::KeyType const hash = hasher(item);
          assert(j == hash);
        }
      }
    }
  }
}
} // namespace hash_table
