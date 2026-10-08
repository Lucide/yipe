#pragma once

#include <vector_types.h>
#include "hashtable/hashtable-common.cuh"
#include "helpers.cuh"
#include "sort/sort-common.cuh"

namespace marching_cubes {

template <typename Hasher>
  requires is_hasher<Hasher, u323>
KERNEL void compute_scalar_fields(float4 const *const points,
                                  HashTable const *const table,
                                  Params const params,
                                  float *const linearized_fields,
                                  Hasher const hasher) {
  using Key = Key<Hasher, float4>;
  u323span const fields(linearized_fields, extents_3d{params.samples.x, params.samples.y, params.samples.z});
  u323 const coords = (vec_cast<u323>(blockIdx) ^ vec_cast<u323>(blockDim)) + vec_cast<u323>(threadIdx);

  if (is_out_of_bounds(coords, params.samples))
    return;

  // convert index to float at maximum precision
  auto const grid_point = get_grid_point(coords, params.particle_diameter);
  float3 const sample_point = clamp_point(grid_point, params.room);

  float field = 0.0f;
  typename Key::KeyType seen_hashes[8];
  u8 seen_count = 0;

  auto const [starts, bounds] = get_neighbourhood_bounds(coords, params.samples);
  for (u83 ind = {}; ind.z < 2; ++ind.z) {
    for (ind.y = 0; ind.y < 2; ++ind.y) {
      for (ind.x = 0; ind.x < 2; ++ind.x) {
        // this style of iterations keeps seen_hashes[] in the registers
        if (!(ind < bounds)) {
          continue;
        }

        s83 const offsets = starts + ind;
        typename Key::KeyType const hash = hasher(coords + offsets);
        // check for clashes in previous hashes
        if (find(seen_hashes, seen_count, hash) < seen_count) {
          continue;
        }
        // keep seen_hashes[] in the registers
        add_checked_hash(seen_hashes, seen_count, hash);

        if (auto const bucket = table->lookup(hash); bucket) {
          for (u32 i = bucket->begin; i < bucket->end; ++i) {
            float3 const point = project4to3(points[i]);
            // currently calculated using Wendland C2
            field += calculate_density(params, sample_point, point);
          }
        }
      }
    }
  }

  field = is_out_of_bounds(grid_point, params.room) ? fminf(field, params.isolevel) : field;
  fields[coords] = field;
}

KERNEL void
count_triangles(Params const params, float const *const linearized_fields, u32 *const linearized_triangle_counts) {
  u323span const fields(linearized_fields, extents_3d{params.samples.x, params.samples.y, params.samples.z});
  u323span const triangle_counts(linearized_triangle_counts,
                                 extents_3d{params.cells.x, params.cells.y, params.cells.z});
  u323 const coords = (vec_cast<u323>(blockIdx) ^ vec_cast<u323>(blockDim)) + vec_cast<u323>(threadIdx);

  if (is_out_of_bounds(coords, params.cells))
    return;

  if (coords == u323{}) {
    // thread 0 writes 0 on an extra cell of triangle counts
    // after the prefix_sum the cell will hold the total number of triangles
    // this allows to store the prefixes on the same buffer as the counts
    linearized_triangle_counts[params.cell_count()] = 0;
  }

  u8 case_index = 0;
  for (u8 i = 0; i < 8; ++i) {
    // read the corner of the cell
    u323 const corner = coords + get_cell_corner_offsets(i);
    // build the corner bitmask (1 means outside the object)
    case_index |= static_cast<u8>(fields[corner] <= params.isolevel) << i;
  }

  triangle_counts[coords] = triangle_table.counts[case_index];
}

KERNEL void make_triangles(Params const params,
                           float const *const linearized_fields,
                           u32 const *const linearized_triangle_prefixes,
                           float33 *const triangles,
                           float33 *const normals) {

  u323span const fields(linearized_fields, extents_3d{params.samples.x, params.samples.y, params.samples.z});
  u323span const triangle_prefixes(linearized_triangle_prefixes,
                                   extents_3d{params.cells.x, params.cells.y, params.cells.z});
  u323 const coord = (vec_cast<u323>(blockIdx) ^ vec_cast<u323>(blockDim)) + vec_cast<u323>(threadIdx);

  if (is_out_of_bounds(coord, params.cells))
    return;

  SampleData sample_data[8];
  u8 case_index = 0;
  for (u8 i = 0; i < 8; ++i) {
    u323 const corner = coord + get_cell_corner_offsets(i);
    sample_data[i].field = fields(corner.x, corner.y, corner.z);
    case_index |= static_cast<u8>(sample_data[i].field <= params.isolevel) << i;
  }

  u8 const triangle_count = triangle_table.counts[case_index];
  if (triangle_count == 0)
    return;

  for (u8 i = 0; i < 8; ++i) {
    u323 const corner = coord + get_cell_corner_offsets(i);
    float3 const f_next = {
        (corner.x + 1 < params.samples.x) ? fields(corner.x + 1, corner.y, corner.z) : params.isolevel,
        (corner.y + 1 < params.samples.y) ? fields(corner.x, corner.y + 1, corner.z) : params.isolevel,
        (corner.z + 1 < params.samples.z) ? fields(corner.x, corner.y, corner.z + 1) : params.isolevel};
    float3 const f_prev = {
        (corner.x > 0) ? fields(corner.x - 1, corner.y, corner.z) : params.isolevel,
        (corner.y > 0) ? fields(corner.x, corner.y - 1, corner.z) : params.isolevel,
        (corner.z > 0) ? fields(corner.x, corner.y, corner.z - 1) : params.isolevel,
    };
    sample_data[i].gradient = f_next - f_prev;
  }

  u32 const start = triangle_prefixes[coord];
  for (u8 i = 0; i < triangle_count; ++i) {
    auto const [p1, gradient1] =
        get_edge_vertex(params, triangle_table.triangles[case_index][i][0], coord, sample_data);
    auto const [p2, gradient2] =
        get_edge_vertex(params, triangle_table.triangles[case_index][i][1], coord, sample_data);
    auto const [p3, gradient3] =
        get_edge_vertex(params, triangle_table.triangles[case_index][i][2], coord, sample_data);

    triangles[start + i] = {
        {p1, p2, p3}
    };
    normals[start + i] = {gradient1, gradient2, gradient3};
  }
}
} // namespace marching_cubes
