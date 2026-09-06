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
  marching_cubes::init_triangle_cases();
  const auto hasher = get_hasher(length, particle_diameter);

  const auto table = build_hashtable(info, points, length, hasher);

  const auto sizes = marching_cubes::Sizes(length, room_dim, particle_diameter, isolevel);
  const dim3 field_block_count = {
      ceil_div<unsigned int, unsigned int, unsigned int>(sizes.fields.x, marching_cubes::threads_per_block.x),
      ceil_div<unsigned int, unsigned int, unsigned int>(sizes.fields.y, marching_cubes::threads_per_block.y),
      ceil_div<unsigned int, unsigned int, unsigned int>(sizes.fields.z, marching_cubes::threads_per_block.z)};

  float *fields;
  handle_cuda_result(cudaMallocAsync(&fields, sizeof(float) * sizes.field_count(), cudaStreamPerThread),
                     "Marching_cubes: allocate scalar fields array");
  handle_cuda_error_debug("Marching_cubes: allocate scalar fields array");
  marching_cubes::compute_scalar_fields<<<field_block_count, marching_cubes::threads_per_block>>>(points, table, sizes,
                                                                                                  fields, hasher);
  handle_cuda_error_debug("Marching_cubes: compute scalar fields");

  // if (sizes.field_count() <= 4096) {
  //   auto cpu_fields = (float *)malloc(sizeof(float) * sizes.field_count());
  //   cudaMemcpy(cpu_fields, fields, sizeof(float) * sizes.field_count(), cudaMemcpyDeviceToHost);
  //   printf("fields:\n");
  //   for (u32 i = 0; i < sizes.field_count(); ++i) {
  //     if (cpu_fields[i] > 0) {
  //       printf("%d:%.1f ", i, cpu_fields[i]);
  //     }
  //   }
  //   printf("\n");
  // }

  const dim3 cell_block_count = {
      ceil_div<unsigned int, unsigned int, unsigned int>(sizes.fields.x, marching_cubes::threads_per_block.x),
      ceil_div<unsigned int, unsigned int, unsigned int>(sizes.fields.y, marching_cubes::threads_per_block.y),
      ceil_div<unsigned int, unsigned int, unsigned int>(sizes.fields.z, marching_cubes::threads_per_block.z)};
  u32 *triangle_counts;
  handle_cuda_result(cudaMallocAsync(&triangle_counts, sizeof(u32) * sizes.cell_count(), cudaStreamPerThread),
                     "Marching_cubes: allocate triangle counts");
  handle_cuda_error_debug("Marching_cubes: allocate triangle counts");
  u8 *triangle_indexes;
  handle_cuda_result(cudaMallocAsync(&triangle_indexes, sizeof(u8) * sizes.cell_count(), cudaStreamPerThread),
                     "Marching_cubes: allocate triangle indexes");
  handle_cuda_error_debug("Marching_cubes: allocate triangle indexes");
  marching_cubes::count_triangles<<<cell_block_count, marching_cubes::threads_per_block>>>(
      sizes, fields, triangle_counts, triangle_indexes, hasher);
  handle_cuda_error_debug("Marching_cubes: count triangles");
  handle_cuda_result(cudaFreeAsync(fields, cudaStreamPerThread), "Marching_cubes: free scalar field array");
  handle_cuda_error_debug("Marching_cubes: free scalar field array");
  handle_cuda_result(cudaFreeAsync(table, cudaStreamPerThread), "Marching_cubes: free hash table");
  handle_cuda_error_debug("Marching_cubes: free hash table");

  u32 *triangle_prefixes;
  handle_cuda_result(cudaMallocAsync(&triangle_prefixes, sizeof(u32) * sizes.cell_count(), cudaStreamPerThread),
                     "Marching_cubes: allocate triangle prefixes");
  handle_cuda_error_debug("Marching_cubes: allocate triangle prefixes");
  scan(triangle_counts, triangle_prefixes, sizes.cell_count());

  u32 max_prefix;
  u32 last_triangles;
  handle_cuda_result(cudaMemcpyAsync(&max_prefix, &triangle_prefixes[sizes.cell_count() - 1], sizeof(u32),
                                     cudaMemcpyDeviceToHost, cudaStreamPerThread),
                     "Marching_cubes: fetch max prefix");
  handle_cuda_error_debug("Marching_cubes: fetch max prefix");
  handle_cuda_result(
      cudaMemcpy(&last_triangles, &triangle_counts[sizes.cell_count() - 1], sizeof(u32), cudaMemcpyDeviceToHost),
      "Marching_cubes: fetch last cell triangle count");
  handle_cuda_error_debug("Marching_cubes: fetch last cell triangle count");

  printf("triangle count: %d\n", max_prefix + last_triangles);
  Triangle *triangles;
  handle_cuda_result(cudaMallocAsync(&triangles, sizeof(Triangle) * (max_prefix + last_triangles), cudaStreamPerThread),
                     "Marching_cubes: allocate triangle buffer");
  handle_cuda_error_debug("Marching_cubes: allocate triangle buffer");

  marching_cubes::make_triangles<<<cell_block_count, marching_cubes::threads_per_block>>>(
      sizes, triangle_counts, triangle_prefixes, triangles, hasher);
  handle_cuda_error_debug("Marching_cubes: make triangles");

  handle_cuda_result(cudaFreeAsync(triangle_counts, cudaStreamPerThread), "Marching_cubes: free triangle counts");
  handle_cuda_error_debug("Marching_cubes: free triangle counts");
  handle_cuda_result(cudaFreeAsync(triangle_indexes, cudaStreamPerThread), "Marching_cubes: free triangle indexes");
  handle_cuda_error_debug("Marching_cubes: free triangle indexes");
  handle_cuda_result(cudaFreeAsync(triangle_prefixes, cudaStreamPerThread), "Marching_cubes: free triangle prefixes");
  handle_cuda_error_debug("Marching_cubes: free triangle prefixes");

  return triangles;
}
