#pragma once

#include "kernels.cuh"

template <typename Data, is_hasher<Data> Hasher>
HashTable *build_hashtable(GPUInfo const &info,
                           Data *const data,
                           u32 const length,
                           u32 const buckets,
                           Hasher const hasher,
                           cudaStream_t const stream) {
  sort(info, data, length, hasher, stream);

  // HashTable - u32 + ((length + 1) * u32)
  u32 const full_table_size = sizeof(HashTable) + sizeof(u32) * buckets;

  HashTable *table;
  handle_cuda_result(cudaMallocAsync(&table, full_table_size, stream), "hashtable: allocate table");
  handle_cuda_error_debug("hashtable: allocate table");
  handle_cuda_result(cudaMemsetAsync(table, 0, full_table_size, stream), "hashtable: memset table");
  handle_cuda_error_debug("hashtable: memset table");
  u32 const blocks_per_grid =
      min(info.multiprocessors * hash_table::blocks_per_sm, ceil_div<u32, u16, u32>(length, hash_table::slice_length));

  hash_table::build<<<blocks_per_grid, hash_table::threads_per_block, 0, stream>>>(data, length, table, buckets,
                                                                                   hasher);
  handle_cuda_error_debug("hashtable: `build` kernel");

  return table;
}

template <typename Data, is_hasher<Data> Hasher>
void verify_hashtable(GPUInfo const &info,
                      Data const *const data,
                      u32 const length,
                      HashTable const *const table,
                      Hasher const hasher,
                      cudaStream_t const stream) {
  u32 const blocks_per_grid = min(info.multiprocessors * hash_table::blocks_per_sm,
                                  (length + hash_table::slice_length - 1) / hash_table::slice_length);
  hash_table::verify<<<blocks_per_grid, hash_table::threads_per_block, 0, stream>>>(data, length, table, hasher);
  handle_cuda_error_debug("hashtable: `verify` kernel");
}
