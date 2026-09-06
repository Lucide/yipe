#pragma once

#include <cuda/std/mdspan>
#include "hashtable/hashtable-common.cuh"
#include "marching-cubes/marching-cubes-common.cuh"
#include "sort/sort-common.cuh"

namespace marching_cubes {
template <typename Hasher>
  requires is_hasher<Hasher, Point>
KERNEL void compute_scalar_fields(const Point *const points,
                                  const HashTable *const table,
                                  const Sizes sizes,
                                  float *const linearized_fields,
                                  const Hasher hasher) {
  using extents_3d =
      cuda::std::extents<u32, cuda::std::dynamic_extent, cuda::std::dynamic_extent, cuda::std::dynamic_extent>;
  cuda::std::mdspan fields(linearized_fields, extents_3d{sizes.fields.x, sizes.fields.y, sizes.fields.z});

  const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
  const u32 y = blockIdx.y * blockDim.y + threadIdx.y;
  const u32 z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= sizes.fields.x || y >= sizes.fields.y || z >= sizes.fields.z)
    return;

  // convert index to float at maximum precision
  const Point node_pos = {static_cast<float>(static_cast<double>(x) * sizes.particle_diameter),
                          static_cast<float>(static_cast<double>(y) * sizes.particle_diameter),
                          static_cast<float>(static_cast<double>(z) * sizes.particle_diameter), 1};

  float scalar_value = 0.0f;

  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const Point neighbor_cell = {node_pos.x + static_cast<float>(dx) * sizes.particle_diameter,
                                     node_pos.y + static_cast<float>(dy) * sizes.particle_diameter,
                                     node_pos.z + static_cast<float>(dz) * sizes.particle_diameter, 1};
        const u32 hash = hasher(neighbor_cell);
        // printf("(%.1f, %.1f, %.1f): %d\n", neighbor_cell.x, neighbor_cell.y, neighbor_cell.z, hash);

        if (const auto bucket = table->lookup(hash); bucket) {
          // 4. Accumulate density from all points in this cell
          for (u32 i = bucket->begin; i < bucket->end; i++) {
            const Point point = points[i];
            scalar_value += calculate_density(node_pos, point, sizes.particle_diameter_sq);
            // printf("%f\n", scalar_value);
          }
        }
      }
    }
  }

  fields(x, y, z) = scalar_value;
}

template <typename Hasher>
  requires is_hasher<Hasher, Point>
KERNEL void count_triangles(const Sizes sizes,
                            const float *const linearized_fields,
                            u32 *const linearized_triangle_counts,
                            u8 *const linearized_triangle_indexes,
                            const Hasher hasher) {
  using extents_3d =
      cuda::std::extents<u32, cuda::std::dynamic_extent, cuda::std::dynamic_extent, cuda::std::dynamic_extent>;
  cuda::std::mdspan fields(linearized_fields, extents_3d{sizes.fields.x, sizes.fields.y, sizes.fields.z});
  cuda::std::mdspan triangle_counts(linearized_triangle_counts,
                                    extents_3d{sizes.cells.x, sizes.cells.y, sizes.cells.z});
  cuda::std::mdspan triangle_indexes(linearized_triangle_indexes,
                                     extents_3d{sizes.cells.x, sizes.cells.y, sizes.cells.z});

  const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
  const u32 y = blockIdx.y * blockDim.y + threadIdx.y;
  const u32 z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= sizes.cells.x || y >= sizes.cells.y || z >= sizes.cells.z)
    return;

  // read the 8 corners of the voxel
  float corners[8];
  for (u8 i = 0; i < 8; ++i) {
    corners[i] = fields(x + cell_corners_offsets[i][0], y + cell_corners_offsets[i][1], z + cell_corners_offsets[i][2]);
  }

  // build the corner bitmask (1 means outside the object)
  u8 cube_index = 0;
  for (u8 i = 0; i < 8; ++i) {
    cube_index |= static_cast<u8>(corners[i] < sizes.isolevel) * (1 << i);
  }

  triangle_indexes(x, y, z) = cube_index;
  triangle_counts(x, y, z) = triangle_table.counts[cube_index];
}

template <typename Hasher>
  requires is_hasher<Hasher, Point>
KERNEL void make_triangles(const Sizes sizes,
                           const u32 *const linearized_triangle_counts,
                           const u32 *const linearized_triangle_prefixes,
                           Triangle *const triangles,
                           const Hasher hasher) {
  using extents_3d =
      cuda::std::extents<u32, cuda::std::dynamic_extent, cuda::std::dynamic_extent, cuda::std::dynamic_extent>;
  cuda::std::mdspan triangle_counts(linearized_triangle_counts,
                                    extents_3d{sizes.cells.x, sizes.cells.y, sizes.cells.z});
  cuda::std::mdspan triangle_prefixes(linearized_triangle_prefixes,
                                      extents_3d{sizes.cells.x, sizes.cells.y, sizes.cells.z});
}
} // namespace marching_cubes
