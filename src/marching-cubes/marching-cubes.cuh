#pragma once

#include <cstdio>
#include <driver_types.h>
#include "hashtable/hashtable.cuh"
#include "kernels.cuh"
#include "scan/scan.cuh"

Triangle *compute_marching_cubes(GPUInfo const *const info,
                                 Point *const points,
                                 const u32 length,
                                 const float3 room_dim,
                                 const float particle_diameter,
                                 const float isolevel) {
  const auto hasher = get_hasher(length, particle_diameter);
  const auto params = marching_cubes::Params(length, room_dim, particle_diameter, isolevel);
  const dim3 field_block_count = {
      ceil_div<unsigned int, unsigned int, unsigned int>(params.fields.x, marching_cubes::threads_per_block.x),
      ceil_div<unsigned int, unsigned int, unsigned int>(params.fields.y, marching_cubes::threads_per_block.y),
      ceil_div<unsigned int, unsigned int, unsigned int>(params.fields.z, marching_cubes::threads_per_block.z)};
  const dim3 cell_block_count = {
      ceil_div<unsigned int, unsigned int, unsigned int>(params.fields.x, marching_cubes::threads_per_block.x),
      ceil_div<unsigned int, unsigned int, unsigned int>(params.fields.y, marching_cubes::threads_per_block.y),
      ceil_div<unsigned int, unsigned int, unsigned int>(params.fields.z, marching_cubes::threads_per_block.z)};

  // compute hashtable
  const auto table = build_hashtable(info, points, length, hasher);

  // compute scalar fields
  float *fields;
  handle_cuda_result(cudaMallocAsync(&fields, sizeof(float) * params.field_count(), cudaStreamPerThread),
                     "Marching_cubes: allocate scalar fields array");
  handle_cuda_error_debug("Marching_cubes: allocate scalar fields array");
  marching_cubes::compute_scalar_fields<<<field_block_count, marching_cubes::threads_per_block>>>(points, table, params,
                                                                                                  fields, hasher);
  handle_cuda_error_debug("Marching_cubes: compute scalar fields");

  // if (params.field_count() <= 4096) {
  //   auto cpu_fields = (float *)malloc(sizeof(float) * params.field_count());
  //   cudaMemcpy(cpu_fields, fields, sizeof(float) * params.field_count(), cudaMemcpyDeviceToHost);
  //   printf("fields:\n");
  //   for (u32 i = 0; i < params.field_count(); ++i) {
  //     if (cpu_fields[i] > 0) {
  //       printf("%d:%.1f ", i, cpu_fields[i]);
  //     }
  //   }
  //   printf("\n");
  // }

  // get triangle counts
  u32 *triangle_counts;
  // we allocate an extra cell that will be set to 0 by the counting kernel
  // after the prefix_sum it will hold the total number of triangles
  // this allows to store the prefixes on the same buffer as the counts
  handle_cuda_result(cudaMallocAsync(&triangle_counts, sizeof(u32) * (params.cell_count() + 1), cudaStreamPerThread),
                     "Marching_cubes: allocate triangle counts");
  handle_cuda_error_debug("Marching_cubes: allocate triangle counts");
  marching_cubes::count_triangles<<<cell_block_count, marching_cubes::threads_per_block>>>(params, fields,
                                                                                           triangle_counts);
  handle_cuda_error_debug("Marching_cubes: count triangles");
  handle_cuda_result(cudaFreeAsync(table, cudaStreamPerThread), "Marching_cubes: free hash table");
  handle_cuda_error_debug("Marching_cubes: free hash table");

  // calculate prefix sum
  scan(triangle_counts, triangle_counts, params.cell_count() + 1);

  // get total triangles
  u32 last_triangles;
  handle_cuda_result(
      cudaMemcpy(&last_triangles, &triangle_counts[params.cell_count()], sizeof(u32), cudaMemcpyDeviceToHost),
      "Marching_cubes: fetch triangle total count");
  handle_cuda_error_debug("Marching_cubes: fetch triangle total count");

  // u32 *count = (u32 *)malloc(sizeof(u32) * (params.cell_count() + 1));
  // cudaMemcpy(count, triangle_counts, sizeof(u32) * (params.cell_count() + 1), cudaMemcpyDeviceToHost);
  // for (u32 i = 0; i < params.cell_count() + 1; ++i) {
  //   printf("%d\n", count[i]);
  // }
  // free(count);

  // make triangles
  printf("triangle count: %d\n", last_triangles);
  Triangle *triangles;
  handle_cuda_result(cudaMallocAsync(&triangles, sizeof(Triangle) * last_triangles, cudaStreamPerThread),
                     "Marching_cubes: allocate triangle buffer");
  handle_cuda_error_debug("Marching_cubes: allocate triangle buffer");

  marching_cubes::make_triangles<<<cell_block_count, marching_cubes::threads_per_block>>>(params, fields,
                                                                                          triangle_counts, triangles);
  handle_cuda_error_debug("Marching_cubes: make triangles");

  handle_cuda_result(cudaFreeAsync(fields, cudaStreamPerThread), "Marching_cubes: free scalar field array");
  handle_cuda_error_debug("Marching_cubes: free scalar field array");
  handle_cuda_result(cudaFreeAsync(triangle_counts, cudaStreamPerThread), "Marching_cubes: free triangle counts");
  handle_cuda_error_debug("Marching_cubes: free triangle counts");

  return triangles;
}
