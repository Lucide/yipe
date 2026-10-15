#include <curand.h>
#include <driver_types.h>
#include "common.h"
#include "hashtable/hashtable.cuh"
#include "marching-cubes/marching-cubes.cuh"
#include "preprocess.cuh"
#include "scan/scan.cuh"
#include "sort/sort.cuh"

// force exporting point_hash even if it is inline
// NOLINTNEXTLINE(readability-redundant-declaration)
extern "C" GPU HOST u32 point_hash(float4 point, u32 particle_count, float particle_diameter);

extern "C" void handle_cuda_result(cudaError_t const err, char const *const message) {
  if (err != cudaSuccess) {
    printf("[%s] CUDA error: %s\n", message, cudaGetErrorString(err));
    exit(-1);
  }
}

extern "C" void handle_cuda_error_debug([[maybe_unused]] char const *const message) {
#ifdef SYNC_ALL
  CPRINT("debug synchronous error handling enabed")
  handle_cuda_result(cudaDeviceSynchronize(), message);
  handle_cuda_result(cudaGetLastError(), message);
#endif
}

extern "C" void
launch_exclusive_scan(u32 const *const src, u32 *const dst, u32 const length, cudaStream_t const stream) {
  scan(src, dst, length, stream);
}

extern "C" void launch_sort_points(GPUInfo const *const info,
                                   float4 *const points,
                                   u32 const length,
                                   float const particle_diameter,
                                   cudaStream_t const stream) {

  auto const hasher = get_hasher(length, particle_diameter);

  sort(*info, points, length, hasher, stream);
}

extern "C" HashTable *new_hashtable(GPUInfo const *const info,
                                    float4 *const points,
                                    u32 const length,
                                    u32 const buckets,
                                    float const particle_diameter,
                                    cudaStream_t const stream) {

  auto const hasher = get_hasher(buckets, particle_diameter);

  return build_hashtable(*info, points, length, buckets, hasher, stream);
}

extern "C" void launch_verify_hashtable(GPUInfo const *const info,
                                        float4 const *const data,
                                        u32 const length,
                                        HashTable const *const table,
                                        float const particle_diameter,
                                        cudaStream_t const stream) {

  u32 table_length;
  TRAP_CUDAPI(cudaMemcpy(&table_length, table, sizeof(u32), cudaMemcpyDeviceToHost),
              "hashtable: download table length");
  auto const hasher = get_hasher(table_length - 1, particle_diameter);

  verify_hashtable(*info, data, length, table, hasher, stream);
}

extern "C" void launch_generate_points(GPUInfo const *const info,
                                       float3 const room,
                                       u32 const seed,
                                       float4 *const out,
                                       u32 const length,
                                       cudaStream_t const stream) {
  float3 *buf;
  TRAP_CUDAPI(cudaMallocAsync(&buf, ARRAY_SIZE(buf, length), stream), "generate_points: allocate random buffer");
  curandGenerator_t gen;
  // initialize cuRAND Host generator
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);
  // curandGenerateUniform4 generates batches of 3 uniform floats natively
  curandGenerateUniform(gen, reinterpret_cast<float *>(buf), length * 3);
  curandDestroyGenerator(gen);
  auto f = [=] GPU(float3 const x) -> float4 { return project3to4(x ^ room); };
  preprocess::preprocess(*info, buf, out, length, f, stream);
  TRAP_CUDAPI(cudaFreeAsync(buf, stream), "generate_points: free random buffer");
}

extern "C" void launch_preprocess_float3(
    GPUInfo const *const info, float3 const *const in, float4 *const out, u32 const length, cudaStream_t const stream) {
  auto f = [] GPU(float3 const x) -> float4 { return project3to4(x); };
  preprocess::preprocess(*info, in, out, length, f, stream);
}

extern "C" float *launch_compute_scalar_fields(GPUInfo const *const info,
                                               float4 *const data,
                                               u32 const length,
                                               float3 const room_dim,
                                               float const particle_diameter,
                                               float const isolevel,
                                               cudaStream_t const stream) {

  return compute_scalar_fields(*info, data, length, room_dim, particle_diameter, isolevel, stream);
}

extern "C" Geometry launch_marching_cubes(float const *const fields,
                                          u32 const length,
                                          float3 const room_dim,
                                          float const particle_diameter,
                                          float const isolevel,
                                          cudaStream_t const stream) {

  return compute_marching_cubes(fields, length, room_dim, particle_diameter, isolevel, stream);
}

extern "C" Geometry launch_triangulate(GPUInfo const *const info,
                                       float4 *const data,
                                       u32 const length,
                                       float3 const room_dim,
                                       float const particle_diameter,
                                       float const isolevel,
                                       cudaStream_t const stream) {

  float const *const fields = compute_scalar_fields(*info, data, length, room_dim, particle_diameter, isolevel, stream);
  return launch_marching_cubes(fields, length, room_dim, particle_diameter, isolevel, stream);
}
