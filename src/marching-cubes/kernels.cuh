#pragma once

#include <vector_types.h>
#include "hashtable/hashtable-common.cuh"
#include "helpers.cuh"
#include "sort/sort-common.cuh"

namespace marching_cubes {

template <typename Hasher>
  requires is_hasher<Hasher, float4>
KERNEL void compute_scalar_fields(float4 const *const points,
                                  HashTable const *const table,
                                  Params const params,
                                  float *const linearized_fields,
                                  Hasher const hasher) {
  using Key = Key<Hasher, float4>;
  cuda::std::mdspan const fields(linearized_fields, extents_3d{params.fields.x, params.fields.y, params.fields.z});
  u323 const coords = (vec_cast(blockIdx) ^ vec_cast(blockDim)) + vec_cast(threadIdx);

  if (is_out_of_bounds(coords, params.fields))
    return;

  // convert index to float at maximum precision
  auto const grid_point = vec_cast<float3>(vec_cast<double3>(coords) * params.particle_diameter);
  float3 const sample_point = clamp_point(grid_point, params.room);

  float scalar_value = 0.0f;
  u32 checked[8];
  u8 checked_count = 0;

  auto const [starts, ends] = get_neighbourhood_offsets(coords, params.fields);
  for (s83 ind = {}; ind.z < 2; ++ind.z) {
    for (ind.y = 0; ind.y < 2; ++ind.y) {
      for (ind.x = 0; ind.x < 2; ++ind.x) {
        // this style of iterations keeps checked[] in the registries
        s83 const offsets = starts + ind;
        if (!(offsets <= ends)) {
          continue;
        }

        // calculate cell position and hash
        float4 const neighbour_cell = project3to4(grid_point + (vec_cast<float3>(offsets) * params.particle_diameter));
        typename Key::KeyType const hash = hasher(neighbour_cell);

        // check for clashes in previous hashes
        if (checked_count > 0 &&
            (checked[0] == hash ||
             (checked_count > 1 &&
              (checked[1] == hash ||
               (checked_count > 2 &&
                (checked[2] == hash ||
                 (checked_count > 3 &&
                  (checked[3] == hash ||
                   (checked_count > 4 &&
                    (checked[4] == hash ||
                     (checked_count > 5 &&
                      (checked[5] == hash ||
                       (checked_count > 6 &&
                        (checked[6] == hash || (checked_count > 7 && checked[7] == hash))))))))))))))) {
          continue;
        }
        checked[checked_count++] = hash; // SPILL

        if (auto const bucket = table->lookup(hash); bucket) {
          for (u32 i = bucket->begin; i < bucket->end; ++i) {
            float3 const point = project4to3(points[i]);
            // currently calculated using Wendland C2
            scalar_value += calculate_density(sample_point, point, params);
          }
        }
      }
    }
  }

  scalar_value = is_out_of_bounds(grid_point, params.room) ? fminf(scalar_value, nextafterf(params.isolevel, -INFINITY))
                                                           : scalar_value;
  fields(coords.x, coords.y, coords.z) = scalar_value;
}

KERNEL void
count_triangles(Params const params, float const *const linearized_fields, u32 *const linearized_triangle_counts) {
  cuda::std::mdspan const fields(linearized_fields, extents_3d{params.fields.x, params.fields.y, params.fields.z});
  cuda::std::mdspan const triangle_counts(linearized_triangle_counts,
                                          extents_3d{params.cells.x, params.cells.y, params.cells.z});
  u323 const coords = (vec_cast(blockIdx) ^ vec_cast(blockDim)) + vec_cast(threadIdx);

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
    float const corner_field = fields(corner.x, corner.y, corner.z);
    // build the corner bitmask (1 means outside the object)
    case_index |= static_cast<u8>(corner_field < params.isolevel) << i;
  }

  triangle_counts(coords.x, coords.y, coords.z) = triangle_table.counts[case_index];
}

KERNEL void make_triangles(Params const params,
                           float const *const linearized_fields,
                           u32 const *const linearized_triangle_prefixes,
                           Triangle *const triangles) {

  cuda::std::mdspan const fields(linearized_fields, extents_3d{params.fields.x, params.fields.y, params.fields.z});
  cuda::std::mdspan const triangle_prefixes(linearized_triangle_prefixes,
                                            extents_3d{params.cells.x, params.cells.y, params.cells.z});
  u323 const coords = (vec_cast(blockIdx) ^ vec_cast(blockDim)) + vec_cast(threadIdx);

  if (is_out_of_bounds(coords, params.cells))
    return;

  // read the 8 corners of the cell
  float corner_fields[8];
  u8 case_index = 0;
  for (u8 i = 0; i < 8; ++i) {
    // read the 8 corners of the cell
    u323 const corner = coords + get_cell_corner_offsets(i);
    corner_fields[i] = fields(corner.x, corner.y, corner.z);
    // build the corner bitmask (1 means outside the object)
    case_index |= static_cast<u8>(corner_fields[i] < params.isolevel) << i;
  }

  u32 const start = triangle_prefixes(coords.x, coords.y, coords.z);
  for (u8 i = 0; i < triangle_table.counts[case_index]; ++i) {
    u8 const e0 = triangle_table.triangles[case_index][i][0];
    u8 const e1 = triangle_table.triangles[case_index][i][1];
    u8 const e2 = triangle_table.triangles[case_index][i][2];

    triangles[start + i] = {
        {get_edge_vertex(params, e0, coords, corner_fields), get_edge_vertex(params, e1, coords, corner_fields),
         get_edge_vertex(params, e2, coords, corner_fields)}
    };
  }
}
} // namespace marching_cubes
