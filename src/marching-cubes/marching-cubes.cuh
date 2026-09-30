#pragma once

#include "hashtable/hashtable.cuh"
#include "kernels.cuh"
#include "scan/scan.cuh"

Geometry compute_marching_cubes(GPUInfo const *const info,
                                float4 *const points,
                                u32 const length,
                                float3 const room,
                                float const particle_diameter,
                                float const isolevel,
                                cudaStream_t const stream) {
  auto const params = marching_cubes::Params(length, room, particle_diameter, isolevel);
  u32 const buckets = std::min(params.cell_count(), length);
  dim3 const field_block_count = {ceil_div<uint, uint, uint>(params.fields.x, marching_cubes::threads_per_block.x),
                                  ceil_div<uint, uint, uint>(params.fields.y, marching_cubes::threads_per_block.y),
                                  ceil_div<uint, uint, uint>(params.fields.z, marching_cubes::threads_per_block.z)};
  dim3 const cell_block_count = {ceil_div<uint, uint, uint>(params.outer_cells.x, marching_cubes::threads_per_block.x),
                                 ceil_div<uint, uint, uint>(params.outer_cells.y, marching_cubes::threads_per_block.y),
                                 ceil_div<uint, uint, uint>(params.outer_cells.z, marching_cubes::threads_per_block.z)};

  // compute hashtable
  auto const hasher = get_hasher(buckets, particle_diameter);
  auto const table = build_hashtable(info, points, length, buckets, hasher, stream);

  // compute scalar fields
  float *fields;
  handle_cuda_result(cudaMallocAsync(&fields, sizeof(float) * params.field_count(), stream),
                     "Marching_cubes: allocate scalar fields array");
  handle_cuda_error_debug("Marching_cubes: allocate scalar fields array");
  marching_cubes::compute_scalar_fields<<<field_block_count, marching_cubes::threads_per_block, 0, stream>>>(
      points, table, params, fields, hasher);
  handle_cuda_error_debug("Marching_cubes: compute scalar fields");

  // get triangle counts
  u32 *triangle_counts;
  // we allocate an extra cell that will be set to 0 by the counting kernel
  // after the prefix_sum it will hold the total number of triangles
  // this allows to store the prefixes on the same buffer as the counts
  handle_cuda_result(cudaMallocAsync(&triangle_counts, sizeof(u32) * (params.outer_cell_count() + 1), stream),
                     "Marching_cubes: allocate triangle counts");
  handle_cuda_error_debug("Marching_cubes: allocate triangle counts");
  marching_cubes::count_triangles<<<cell_block_count, marching_cubes::threads_per_block, 0, stream>>>(params, fields,
                                                                                                      triangle_counts);
  handle_cuda_error_debug("Marching_cubes: count triangles");
  handle_cuda_result(cudaFreeAsync(table, stream), "Marching_cubes: free hash table");
  handle_cuda_error_debug("Marching_cubes: free hash table");

  // calculate prefix sum
  scan(triangle_counts, triangle_counts, params.outer_cell_count() + 1, stream);

  // get total triangles
  u32 total_triangles;
  handle_cuda_result(cudaMemcpyAsync(&total_triangles, &triangle_counts[params.outer_cell_count()], sizeof(u32),
                                     cudaMemcpyDeviceToHost, stream),
                     "Marching_cubes: fetch triangle total count");
  handle_cuda_result(cudaStreamSynchronize(stream), "Marching_cubes: fetch triangle total count");
  handle_cuda_error_debug("Marching_cubes: fetch triangle total count");

  // make triangles
  Triangle *triangles;
  handle_cuda_result(cudaMallocAsync(&triangles, sizeof(Triangle) * total_triangles, stream),
                     "Marching_cubes: allocate triangle buffer");
  handle_cuda_error_debug("Marching_cubes: allocate triangle buffer");

  marching_cubes::make_triangles<<<cell_block_count, marching_cubes::threads_per_block, 0, stream>>>(
      params, fields, triangle_counts, triangles);
  handle_cuda_error_debug("Marching_cubes: make triangles");

  handle_cuda_result(cudaFreeAsync(fields, stream), "Marching_cubes: free scalar field array");
  handle_cuda_error_debug("Marching_cubes: free scalar field array");
  handle_cuda_result(cudaFreeAsync(triangle_counts, stream), "Marching_cubes: free triangle counts");
  handle_cuda_error_debug("Marching_cubes: free triangle counts");

  return {total_triangles, triangles};
}
