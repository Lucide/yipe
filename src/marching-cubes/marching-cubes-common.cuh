#pragma once
#include <cuda/std/mdspan>
#include <cuda_runtime.h>
#include "gpu-common.cuh"
#include "tables.cuh"

namespace marching_cubes {

using extents_3d =
    cuda::std::extents<u32, cuda::std::dynamic_extent, cuda::std::dynamic_extent, cuda::std::dynamic_extent>;

constexpr dim3 threads_per_block = {8, 8, 8};

struct Params {
  u32 points;
  float3 room_dim{};
  float particle_diameter;
  float particle_diameter_sq;
  uint3 cells{};
  uint3 fields{};
  float isolevel;

  Params(const u32 points, const float3 room_dim, const float particle_diameter, const float isolevel) {
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
  GPU HOST constexpr u32 cell_count() const { return this->cells.x * this->cells.y * this->cells.z; }
  GPU HOST constexpr u32 field_count() const { return this->fields.x * this->fields.y * this->fields.z; }
};

GPU float calculate_density(const Point node_pos, const Point point_pos, const float radius_sq) {
  const float dist_sq = sq_dist(node_pos, point_pos);

  if (dist_sq >= radius_sq) {
    return 0.0f;
  }

  // Quadratic falloff kernel (Metaball style)
  float v = 1.0f / dist_sq;
  return v;
}

GPU Point interpolate_vertex(const float isolevel, Point p1, Point p2, const float field1, const float field2) {
  if (p2 < p1) {
    cuda::std::swap(p1, p2);
  }

  Point p;
  if (fabs(field1 - field2) > 0.00001)
    p = p1 + (p2 - p1) / (field2 - field1) * (isolevel - field1);
  else
    p = p1;
  return p;
}

GPU Point INLINE get_cell_corner(const uint3 cell, const u8 corner, const float particle_diameter) {
  const Offsets offsets = get_cell_corner_offsets(corner);
  return {static_cast<float>(static_cast<double>(cell.x + offsets.x) * particle_diameter),
          static_cast<float>(static_cast<double>(cell.y + offsets.y) * particle_diameter),
          static_cast<float>(static_cast<double>(cell.z + offsets.z) * particle_diameter), 1};
}

GPU INLINE float select_field(const u8 corner, const float *const fields) {
  switch (corner) {
  case 0:
    return fields[0];
  case 1:
    return fields[1];
  case 2:
    return fields[2];
  case 3:
    return fields[3];
  case 4:
    return fields[4];
  case 5:
    return fields[5];
  case 6:
    return fields[6];
  case 7:
    return fields[7];
  default:
    assert(corner < 8);
    return 0.0f;
  }
}

GPU Point INLINE get_edge_vertex(
    const u8 edge, const float isolevel, const uint3 cell, const float particle_diameter, const float *const fields) {
  const EdgeVertexes vertexes = get_edge_vertexes(edge);

  const Point corner0 = get_cell_corner(cell, vertexes.c0, particle_diameter);
  const Point corner1 = get_cell_corner(cell, vertexes.c1, particle_diameter);

  float field0 = select_field(vertexes.c0, fields);

  float field1 = select_field(vertexes.c0, fields);

  return interpolate_vertex(isolevel, corner0, corner1, field0, field1);
}
} // namespace marching_cubes
