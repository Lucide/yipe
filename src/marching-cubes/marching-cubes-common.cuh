#pragma once
#include <cstdio>
#include <cuda_runtime.h>
#include "gpu-common.cuh"
#include "table.cuh"

namespace marching_cubes {

constexpr dim3 threads_per_block = {8, 8, 8};

struct Sizes {
  u32 points;
  float3 room_dim{};
  float particle_diameter;
  float particle_diameter_sq;
  uint3 cells{};
  uint3 fields{};
  float isolevel;

  Sizes(const u32 points, const float3 room_dim, const float particle_diameter, const float isolevel) {
    this->points = points;
    this->room_dim = room_dim;
    this->particle_diameter = particle_diameter;
    this->particle_diameter_sq = particle_diameter * particle_diameter;
    this->isolevel = isolevel;
    this->cells = {static_cast<u32>(ceil(room_dim.x / particle_diameter)),
                   static_cast<u32>(ceil(room_dim.y / particle_diameter)),
                   static_cast<u32>(ceil(room_dim.z / particle_diameter))};
    this->fields = {this->cells.x + 1, this->cells.y + 1, this->cells.z + 1};
  }
  u32 cell_count() const { return this->cells.x * this->cells.y * this->cells.z; }
  u32 field_count() const { return this->fields.x * this->fields.y * this->fields.z; }
};

[[maybe_unused]] CONSTANT TriangleCases triangle_table = {};

void init_triangle_cases() {
  // init the cases only once
  static bool done = false;
  if (!done) {
    // printf("count init list: {");
    // for (u32 i = 0; i < 256; ++i) {
    //   u32 count = 0;
    //   for (u32 j = 0; j < 5; ++j) {
    //     if (static_table.triangles[i][j][0] != -1) {
    //       ++count;
    //     }
    //   }
    //   printf("%d, ", count);
    // }
    // printf("}\n");
    done = true;
    cudaMemcpyToSymbolAsync(triangle_table, &static_table, sizeof(TriangleCases));
  }
}

GPU float calculate_density(const Point node_pos, const Point point_pos, const float radius_sq) {
  const float dist_sq = sq_dist(node_pos, point_pos);

  if (dist_sq >= radius_sq) {
    return 0.0f;
  }
  // printf("(%.1f, %.1f, %.1f) (%.1f, %.1f, %.1f)\n", node_pos.x, node_pos.y, node_pos.z, point_pos.x, point_pos.y,
  //        point_pos.z);
  // printf("%f %f\n", dist_sq, radius_sq);

  // Quadratic falloff kernel (Metaball style)
  float v = 1.0f / dist_sq;
  return v;
}
} // namespace marching_cubes
