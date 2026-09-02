#pragma once

#include "hashtable/hashtable-common.cuh"
#include "sort/sort.cuh"

namespace hash_table {
inline constexpr u32 slice_length = threads_per_block * items_per_thread_per_slice;
struct HashTable {
  u32 *table;
};

template <typename Data, typename Hasher>
// requires is_hasher<Hasher, Data>
KERNEL void find(Data *const data, u32 const length, HashTable table, Hasher hasher) {}

} // namespace hash_table

template <typename Data, typename Hasher>
hash_table::HashTable hashtable(GPUInfo const *const info, Data *const data, u32 const length, Hasher hasher) {
  sort(info, data, length, hasher);

  const u32 table_size = sizeof(u32) * length + 1;

  hash_table::HashTable table{};
  cudaMalloc(&table.table, table_size);
  cudaMemset(table.table, 0, table_size);
  u32 const blocks_per_grid = min(info->multiprocessors * hash_table::blocks_per_sm,
                                  (length + hash_table::slice_length - 1) / hash_table::slice_length);

  hash_table::find<<<blocks_per_grid, hash_table::threads_per_block>>>(data, length, table, hasher);
}
