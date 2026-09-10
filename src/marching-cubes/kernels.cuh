#pragma once

#include "hashtable/hashtable-common.cuh"
#include "marching-cubes-common.cuh"
#include "sort/sort-common.cuh"

namespace marching_cubes {
template <typename Hasher>
  requires is_hasher<Hasher, Point>
KERNEL void compute_scalar_fields(const Point *const points,
                                  const HashTable *const table,
                                  const Params params,
                                  float *const linearized_fields,
                                  const Hasher hasher) {
  using Key = Key<Hasher, Point>;
  const cuda::std::mdspan fields(linearized_fields, extents_3d{params.fields.x, params.fields.y, params.fields.z});

  const uint3 vertex = {
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y,
      blockIdx.z * blockDim.z + threadIdx.z,
  };
  if (vertex.x >= params.fields.x || vertex.y >= params.fields.y || vertex.z >= params.fields.z)
    return;

  // convert index to float at maximum precision
  const Point vertex_pos = {static_cast<float>(static_cast<double>(vertex.x) * params.particle_diameter),
                            static_cast<float>(static_cast<double>(vertex.y) * params.particle_diameter),
                            static_cast<float>(static_cast<double>(vertex.z) * params.particle_diameter), 1};

  float scalar_value = 0.0f;
  u32 checked[8];
  u8 i = 0;
  for (i8 dz = -1; dz <= 0; ++dz) {
    for (i8 dy = -1; dy <= 0; ++dy) {
      for (i8 dx = -1; dx <= 0; ++dx, i += 1) {
        // calculate cell position and hash
        const Point neighbour_cell = {vertex_pos.x + static_cast<float>(dx) * params.particle_diameter,
                                      vertex_pos.y + static_cast<float>(dy) * params.particle_diameter,
                                      vertex_pos.z + static_cast<float>(dz) * params.particle_diameter, 1};
        const typename Key::KeyType hash = hasher(neighbour_cell);
        // add hash
        checked[i] = hash;

        // check if the cell is outside of the room's bounds
        const int3 neighbour = {
            static_cast<int>(vertex.x) + dx,
            static_cast<int>(vertex.y) + dy,
            static_cast<int>(vertex.z) + dz,
        };
        if (neighbour.x < 0 || neighbour.y < 0 || neighbour.z < 0 || neighbour.x >= static_cast<int>(params.cells.x) ||
            neighbour.y >= static_cast<int>(params.cells.y) || neighbour.z >= static_cast<int>(params.cells.z)) {
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

        if (const auto bucket = table->lookup(hash); bucket) {
          for (u32 i = bucket->begin; i < bucket->end; i++) {
            const Point point = points[i];
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
count_triangles(const Params params, const float *const linearized_fields, u32 *const linearized_triangle_counts) {
  const cuda::std::mdspan fields(linearized_fields, extents_3d{params.fields.x, params.fields.y, params.fields.z});
  const cuda::std::mdspan triangle_counts(linearized_triangle_counts,
                                          extents_3d{params.cells.x, params.cells.y, params.cells.z});

  const uint3 cell = {
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y,
      blockIdx.z * blockDim.z + threadIdx.z,
  };
  if (cell.x >= params.cells.x || cell.y >= params.cells.y || cell.z >= params.cells.z)
    return;

  if (cell.x == 0 && cell.y == 0 && cell.z == 0) {
    // thread 0 writes 0 onan extra cell of triangle counts
    // after the prefix_sum the cell will hold the total number of triangles
    // this allows to store the prefixes on the same buffer as the counts
    linearized_triangle_counts[params.cell_count()] = 0;
  }

  // read the 8 corners of the voxel
  float corner_fields[8];
  for (u8 i = 0; i < 8; ++i) {
    const Offsets offsets = get_cell_corner_offsets(i);
    corner_fields[i] = fields(cell.x + offsets.x, cell.y + offsets.y, cell.z + offsets.z);
  }
  // build the corner bitmask (1 means outside the object)
  u8 cube_index = 0;
  for (u8 i = 0; i < 8; ++i) {
    cube_index |= static_cast<u8>(corner_fields[i] < params.isolevel) * (1 << i);
  }

  triangle_counts(cell.x, cell.y, cell.z) = triangle_table.counts[cube_index];
}

KERNEL void make_triangles(const Params params,
                           const float *const linearized_fields,
                           const u32 *const linearized_triangle_prefixes,
                           Triangle *const triangles) {

  const cuda::std::mdspan fields(linearized_fields, extents_3d{params.fields.x, params.fields.y, params.fields.z});
  const cuda::std::mdspan triangle_prefixes(linearized_triangle_prefixes,
                                            extents_3d{params.cells.x, params.cells.y, params.cells.z});

  const uint3 cell = {
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y,
      blockIdx.z * blockDim.z + threadIdx.z,
  };
  if (cell.x >= params.cells.x || cell.y >= params.cells.y || cell.z >= params.cells.z)
    return;

  // read the 8 corners of the voxel
  float corner_fields[8];
  for (u8 i = 0; i < 8; ++i) {
    const Offsets offsets = get_cell_corner_offsets(i);
    corner_fields[i] = fields(cell.x + offsets.x, cell.y + offsets.y, cell.z + offsets.z);
  }
  // build the corner bitmask (1 means outside the object)
  u8 cube_index = 0;
  for (u8 i = 0; i < 8; ++i) {
    cube_index |= static_cast<u8>(corner_fields[i] < params.isolevel) * (1 << i);
  }

  const u32 start = triangle_prefixes(cell.x, cell.y, cell.z);
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
