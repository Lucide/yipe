#pragma once

#include "hashtable/hashtable.cuh"
#include "kernels.cuh"
#include "scan/scan.cuh"

float *compute_scalar_fields(GPUInfo const &info,
                             float4 *const points,
                             u32 const length,
                             float3 const room,
                             float const particle_diameter,
                             float const isolevel,
                             cudaStream_t const stream) {
  auto const params = marching_cubes::Params(length, room, particle_diameter, isolevel);
  u32 const buckets = std::min(params.cell_count(), length);
  dim3 const field_block_count = {ceil_div<uint, uint, uint>(params.samples.x, marching_cubes::threads_per_block.x),
                                  ceil_div<uint, uint, uint>(params.samples.y, marching_cubes::threads_per_block.y),
                                  ceil_div<uint, uint, uint>(params.samples.z, marching_cubes::threads_per_block.z)};

  // compute hashtable
  auto const hasher = get_hasher(buckets, particle_diameter);
  auto const table = build_hashtable(info, points, length, buckets, hasher, stream);

  // compute scalar fields
  float *fields;
  TRAP_CUDAPI(cudaMallocAsync(&fields, ARRAY_SIZE(fields, params.samples_count()), stream),
              "marching_cubes: allocate scalar fields array");
  TRAP_KERNEL(
      (marching_cubes::compute_scalar_fields<<<field_block_count, marching_cubes::threads_per_block, 0, stream>>>(
          points, table, params, fields, hasher)),
      "marching_cubes: compute scalar fields");
  TRAP_CUDAPI(cudaFreeAsync(table, stream), "marching_cubes: free hash table");
  return fields;
}

Geometry compute_marching_cubes(float const *const fields,
                                u32 const length,
                                float3 const room,
                                float const particle_diameter,
                                float const isolevel,
                                cudaStream_t const stream) {
  auto const params = marching_cubes::Params(length, room, particle_diameter, isolevel);
  dim3 const cell_block_count = {ceil_div<uint, uint, uint>(params.cells.x, marching_cubes::threads_per_block.x),
                                 ceil_div<uint, uint, uint>(params.cells.y, marching_cubes::threads_per_block.y),
                                 ceil_div<uint, uint, uint>(params.cells.z, marching_cubes::threads_per_block.z)};

  // get triangle counts
  u32 *triangle_counts;
  u32 const triangle_counts_length = params.cell_count() + 1;
  // we allocate an extra cell that will be set to 0 by the counting kernel
  // after the prefix_sum it will hold the total number of triangles
  // this allows to store the prefixes on the same buffer as the counts
  TRAP_CUDAPI(cudaMallocAsync(&triangle_counts, ARRAY_SIZE(triangle_counts, triangle_counts_length), stream),
              "marching_cubes: allocate triangle counts");
  TRAP_KERNEL((marching_cubes::count_triangles<<<cell_block_count, marching_cubes::threads_per_block, 0, stream>>>(
                  params, fields, triangle_counts)),
              "marching_cubes: count triangles");

  // calculate prefix sum
  scan(triangle_counts, triangle_counts, triangle_counts_length, stream);

  // get total triangles
  u32 total_triangles;
  TRAP_CUDAPI(cudaMemcpy(&total_triangles, &triangle_counts[params.cell_count()], sizeof(u32), cudaMemcpyDeviceToHost),
              "marching_cubes: fetch triangle total count");

  // make triangles
  float33 *triangles;
  TRAP_CUDAPI(cudaMallocAsync(&triangles, ARRAY_SIZE(triangles, 2 * total_triangles), stream),
              "marching_cubes: allocate triangle buffer");
  float33 *const normals = triangles + total_triangles;

  TRAP_KERNEL((marching_cubes::make_triangles<<<cell_block_count, marching_cubes::threads_per_block, 0, stream>>>(
                  params, fields, triangle_counts, triangles, normals)),
              "marching_cubes: make triangles");

  TRAP_CUDAPI(cudaFreeAsync(const_cast<float *>(fields), stream), "marching_cubes: free scalar field array");
  TRAP_CUDAPI(cudaFreeAsync(triangle_counts, stream), "marching_cubes: free triangle counts");

  return {triangles, normals, total_triangles, {}};
}
