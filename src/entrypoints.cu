#include "common.h"
#include "hashtable/hashtable.cuh"
#include "marching-cubes/marching-cubes.cuh"
#include "scan/scan.cuh"
#include "sort/sort.cuh"

extern "C" void handle_cuda_result(const cudaError_t err, const char *message) {
  if (err != cudaSuccess) {
    printf("[%s] CUDA Error: %s\n", message, cudaGetErrorString(err));
    exit(-1);
  }
}

extern "C" void handle_cuda_error(const char *message) {
  handle_cuda_result(cudaDeviceSynchronize(), message);
  handle_cuda_result(cudaGetLastError(), message);
}

extern "C" void handle_cuda_error_debug([[maybe_unused]] const char *message) {
#ifndef NDEBUG
  CPRINT("Debug synchronous error handling enabed")
  handle_cuda_result(cudaDeviceSynchronize(), message);
  handle_cuda_result(cudaGetLastError(), message);
#endif
}

extern "C" void launch_exclusive_scan(const u32 *const src, u32 *const dst, u32 const length) {
  scan(src, dst, length);
}

extern "C" void launch_sort_points(GPUInfo const *const info, Point *const points, u32 const length) {

  const auto hasher = get_hasher(length, PARTICLE_DIAMETER);

  sort(info, points, length, hasher);
}

extern "C" HashTable *new_hashtable(GPUInfo const *const info, Point *const points, u32 const length) {

  const auto hasher = get_hasher(length, PARTICLE_DIAMETER);

  return build_hashtable(info, points, length, hasher);
}

extern "C" void launch_verify_hashtable(GPUInfo const *const info,
                                        const Point *const data,
                                        u32 const length,
                                        const HashTable *const table) {

  const auto hasher = get_hasher(length, PARTICLE_DIAMETER);

  verify_hashtable(info, data, length, table, hasher);
}

extern "C" Triangle *triangulate(GPUInfo const *const info,
                                 Point *const data,
                                 const u32 length,
                                 const float3 room_dim,
                                 const float particle_diameter,
                                 const float isolevel) {

  return compute_marching_cubes(info, data, length, room_dim, particle_diameter, isolevel);
}
