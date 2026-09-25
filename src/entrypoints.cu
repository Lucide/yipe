#include <driver_types.h>
#include "common.h"
#include "hashtable/hashtable.cuh"
#include "marching-cubes/marching-cubes.cuh"
#include "scan/scan.cuh"
#include "sort/sort.cuh"

extern "C" void handle_cuda_result(cudaError_t const err, char const *message) {
  if (err != cudaSuccess) {
    printf("[%s] CUDA Error: %s\n", message, cudaGetErrorString(err));
    exit(-1);
  }
}

extern "C" void handle_cuda_error(char const *message) {
  handle_cuda_result(cudaDeviceSynchronize(), message);
  handle_cuda_result(cudaGetLastError(), message);
}

extern "C" void handle_cuda_error_debug([[maybe_unused]] char const *message) {
#ifndef NDEBUG
  CPRINT("Debug synchronous error handling enabed")
  handle_cuda_result(cudaDeviceSynchronize(), message);
  handle_cuda_result(cudaGetLastError(), message);
#endif
}

extern "C" void launch_exclusive_scan(u32 const *const src, u32 *const dst, u32 const length) {
  scan(src, dst, length);
}

extern "C" void launch_sort_points(GPUInfo const *const info, float4 *const points, u32 const length) {

  auto const hasher = get_hasher(length, PARTICLE_DIAMETER);

  sort(info, points, length, hasher);
}

extern "C" HashTable *
new_hashtable(GPUInfo const *const info, float4 *const points, u32 const length, u32 const buckets) {

  auto const hasher = get_hasher(buckets, PARTICLE_DIAMETER);

  return build_hashtable(info, points, length, buckets, hasher);
}

extern "C" void launch_verify_hashtable(GPUInfo const *const info,
                                        float4 const *const data,
                                        u32 const length,
                                        HashTable const *const table) {

  u32 table_length;
  cudaMemcpy(&table_length, table, sizeof(u32), cudaMemcpyDeviceToHost);
  auto const hasher = get_hasher(table_length - 1, PARTICLE_DIAMETER);

  verify_hashtable(info, data, length, table, hasher);
}

extern "C" Geometry triangulate(GPUInfo const *const info,
                                float4 *const data,
                                u32 const length,
                                float3 const room_dim,
                                float const particle_diameter,
                                float const isolevel) {

  return compute_marching_cubes(info, data, length, room_dim, particle_diameter, isolevel);
}
