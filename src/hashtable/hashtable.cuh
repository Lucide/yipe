#pragma once

#include "hashtable/kernels.cuh"
#include "kernels.cuh"

template <typename Data, typename Hasher>
HashTable *new_hashtable(GPUInfo const *const info, Data *const data, u32 const length, Hasher hasher) {
  sort(info, data, length, hasher);

  const u32 full_table_size = sizeof(HashTable) + sizeof(u32) * length + 1 - 1;

  HashTable *table;
  handle_cuda_result(cudaMalloc(&table, full_table_size), "Hashtable: Allocate table");
  handle_cuda_result(cudaMemset(table, 0, full_table_size), "Hashtable: Memset table");
  u32 const blocks_per_grid = min(info->multiprocessors * hash_table::blocks_per_sm,
                                  (length + hash_table::slice_length - 1) / hash_table::slice_length);

  hash_table::build<<<blocks_per_grid, hash_table::threads_per_block>>>(data, length, table, hasher);
  handle_cuda_error("Hashtable: `build` kernel");

  return table;
}
