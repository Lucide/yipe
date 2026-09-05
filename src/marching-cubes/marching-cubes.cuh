#pragma once

#include <cuda_runtime.h>
#include "hashtable/hashtable.cuh"
#include "marching-cubes/kernels.cuh"

void triangulate(GPUInfo const *const info,
                 Point *const points,
                 const u32 length,
                 const float3 room_dim,
                 const float particle_diameter) {
  const auto hasher = get_hasher(length, particle_diameter);
  const auto table = new_hashtable(info, points, length, hasher);
  const auto sizes = marching_cubes::Sizes(length, room_dim, particle_diameter);
  float *fields;
  cudaMallocAsync(&fields, sizeof(float) * sizes.fields_count(), cudaStreamPerThread);
  marching_cubes::compute_scalar_fields<<<1, 1>>>(points, table, sizes, fields);
}
