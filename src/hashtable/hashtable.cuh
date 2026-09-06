#pragma once

#include "hashtable/kernels.cuh"
#include "kernels.cuh"

template <typename Data, typename Hasher>
  requires is_hasher<Hasher, Data>
HashTable *build_hashtable(GPUInfo const *const info, Data *const data, u32 const length, const Hasher hasher) {
  sort(info, data, length, hasher);

  // HashTable - u32 + ((length + 1) * u32)
  const u32 full_table_size = sizeof(HashTable) + sizeof(u32) * length;

  HashTable *table;
  handle_cuda_result(cudaMallocAsync(&table, full_table_size, cudaStreamPerThread), "Hashtable: Allocate table");
  handle_cuda_error_debug("Hashtable: Allocate table");
  handle_cuda_result(cudaMemsetAsync(table, 0, full_table_size), "Hashtable: Memset table");
  handle_cuda_error_debug("Hashtable: Memset table");
  u32 const blocks_per_grid =
      min(info->multiprocessors * hash_table::blocks_per_sm, ceil_div<u32, u16, u32>(length, hash_table::slice_length));

  hash_table::build<<<blocks_per_grid, hash_table::threads_per_block>>>(data, length, table, hasher);
  handle_cuda_error_debug("Hashtable: `build` kernel");

  return table;
}

template <typename Data, typename Hasher>
  requires is_hasher<Hasher, Data>
void verify_hashtable(GPUInfo const *const info,
                      const Data *const data,
                      u32 const length,
                      const HashTable *const table,
                      const Hasher hasher) {
  u32 const blocks_per_grid = min(info->multiprocessors * hash_table::blocks_per_sm,
                                  (length + hash_table::slice_length - 1) / hash_table::slice_length);
  hash_table::verify<<<blocks_per_grid, hash_table::threads_per_block>>>(data, length, table, hasher);
  handle_cuda_error_debug("Hashtable: `verify` kernel");
}
