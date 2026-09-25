#pragma once

#include "hashtable/hashtable-common.cuh"
#include "marching-cubes-common.cuh"
#include "sort/sort-common.cuh"

namespace marching_cubes {
template <typename Hasher>
  requires is_hasher<Hasher, Point>
KERNEL void compute_scalar_fields(Point const *const points,
                                  HashTable const *const table,
                                  Params const params,
                                  float *const linearized_fields,
                                  Hasher const hasher) {
  using Key = Key<Hasher, Point>;
  cuda::std::mdspan const fields(linearized_fields, extents_3d{params.fields.x, params.fields.y, params.fields.z});

  uint3 const vertex = {
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y,
      blockIdx.z * blockDim.z + threadIdx.z,
  };
  if (vertex.x >= params.fields.x || vertex.y >= params.fields.y || vertex.z >= params.fields.z)
    return;

  // convert index to float at maximum precision
  Point const vertex_pos = {dtof(static_cast<double>(vertex.x) * params.particle_diameter),
                            dtof(static_cast<double>(vertex.y) * params.particle_diameter),
                            dtof(static_cast<double>(vertex.z) * params.particle_diameter), 1};

  float scalar_value = 0.0f;
  u32 checked[8];
  u8 i = 0;
  for (i8 dz = -1; dz <= 0; ++dz) {
    for (i8 dy = -1; dy <= 0; ++dy) {
      for (i8 dx = -1; dx <= 0; ++dx, i += 1) {
        // calculate cell position and hash
        Point const neighbour_cell = {vertex_pos.x + itof(dx) * params.particle_diameter,
                                      vertex_pos.y + itof(dy) * params.particle_diameter,
                                      vertex_pos.z + itof(dz) * params.particle_diameter, 1};
        typename Key::KeyType const hash = hasher(neighbour_cell);
        // add hash
        checked[i] = hash;

        // check if the cell is outside of the room's bounds
        int3 const neighbour = {
            utos(vertex.x) + dx,
            utos(vertex.y) + dy,
            utos(vertex.z) + dz,
        };
        if (neighbour.x < 0 || neighbour.y < 0 || neighbour.z < 0 || neighbour.x >= utos(params.cells.x) ||
            neighbour.y >= utos(params.cells.y) || neighbour.z >= utos(params.cells.z)) {
          continue;
        }

        // check for clashes in previous hashes
        if (i > 0 &&
            (checked[0] == hash ||
             (i > 1 &&
              (checked[1] == hash ||
               (i > 2 && (checked[2] == hash ||
                          (i > 3 && (checked[3] == hash ||
                                     (i > 4 && (checked[4] == hash ||
                                                (i > 5 && (checked[5] == hash ||
                                                           (i > 6 && (checked[6] == hash ||
                                                                      (i > 7 && checked[7] == hash))))))))))))))) {
          continue;
        }

        if (auto const bucket = table->lookup(hash); bucket) {
          for (u32 i = bucket->begin; i < bucket->end; i++) {
            Point const point = points[i];
            // currently calculated using Wendland C2
            scalar_value += calculate_density(vertex_pos, point, params);
          }
        }
      }
    }
  }

  fields(vertex.x, vertex.y, vertex.z) = scalar_value;
}

KERNEL void
count_triangles(Params const params, float const *const linearized_fields, u32 *const linearized_triangle_counts) {
  cuda::std::mdspan const fields(linearized_fields, extents_3d{params.fields.x, params.fields.y, params.fields.z});
  cuda::std::mdspan const triangle_counts(linearized_triangle_counts,
                                          extents_3d{params.cells.x, params.cells.y, params.cells.z});

  uint3 const cell = {
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y,
      blockIdx.z * blockDim.z + threadIdx.z,
  };
  if (cell.x >= params.cells.x || cell.y >= params.cells.y || cell.z >= params.cells.z)
    return;

  if (cell.x == 0 && cell.y == 0 && cell.z == 0) {
    // thread 0 writes 0 on an extra cell of triangle counts
    // after the prefix_sum the cell will hold the total number of triangles
    // this allows to store the prefixes on the same buffer as the counts
    linearized_triangle_counts[params.cell_count()] = 0;
  }

  u8 cube_index = 0;
  for (u8 i = 0; i < 8; ++i) {
    // read the corner of the voxel
    Offsets const offsets = get_cell_corner_offsets(i);
    int3 const corner = {utos(cell.x + offsets.x), utos(cell.y + offsets.y), utos(cell.z + offsets.z)};
    float const corner_field = fields(corner.x, corner.y, corner.z);

    // build the corner bitmask (1 means outside the object)
    bool const wall = corner.x == 0 || corner.y == 0 || corner.z == 0 || corner.x == utos(params.fields.x - 1) ||
                      corner.y == utos(params.fields.y - 1) || corner.z == utos(params.fields.z - 1);
    cube_index |= static_cast<u8>(corner_field < params.isolevel || wall) * (1 << i);
  }

  triangle_counts(cell.x, cell.y, cell.z) = triangle_table.counts[cube_index];
}

KERNEL void make_triangles(Params const params,
                           float const *const linearized_fields,
                           u32 const *const linearized_triangle_prefixes,
                           Triangle *const triangles) {

  cuda::std::mdspan const fields(linearized_fields, extents_3d{params.fields.x, params.fields.y, params.fields.z});
  cuda::std::mdspan const triangle_prefixes(linearized_triangle_prefixes,
                                            extents_3d{params.cells.x, params.cells.y, params.cells.z});

  uint3 const cell = {
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y,
      blockIdx.z * blockDim.z + threadIdx.z,
  };
  if (cell.x >= params.cells.x || cell.y >= params.cells.y || cell.z >= params.cells.z)
    return;

  // read the 8 corners of the voxel
  float corner_fields[8];
  u8 cube_index = 0;
  for (u8 i = 0; i < 8; ++i) {
    // read the 8 corners of the voxel
    Offsets const offsets = get_cell_corner_offsets(i);
    int3 const corner = {utos(cell.x + offsets.x), utos(cell.y + offsets.y), utos(cell.z + offsets.z)};
    corner_fields[i] = fields(corner.x, corner.y, corner.z);

    // build the corner bitmask (1 means outside the object)
    bool const wall = corner.x == 0 || corner.y == 0 || corner.z == 0 || corner.x == utos(params.fields.x - 1) ||
                      corner.y == utos(params.fields.y - 1) || corner.z == utos(params.fields.z - 1);
    cube_index |= static_cast<u8>(corner_fields[i] < params.isolevel || wall) * (1 << i);
  }

  u32 const start = triangle_prefixes(cell.x, cell.y, cell.z);
  for (u8 i = 0; i < triangle_table.counts[cube_index]; ++i) {
    u8 e0 = static_cast<u8>(triangle_table.triangles[cube_index][i][0]);
    u8 e1 = static_cast<u8>(triangle_table.triangles[cube_index][i][1]);
    u8 e2 = static_cast<u8>(triangle_table.triangles[cube_index][i][2]);

    triangles[start + i].points[0] =
        project4to3(get_edge_vertex(e0, params.isolevel, cell, params.particle_diameter, corner_fields));
    triangles[start + i].points[1] =
        project4to3(get_edge_vertex(e1, params.isolevel, cell, params.particle_diameter, corner_fields));
    triangles[start + i].points[2] =
        project4to3(get_edge_vertex(e2, params.isolevel, cell, params.particle_diameter, corner_fields));
  }
}
} // namespace marching_cubes
