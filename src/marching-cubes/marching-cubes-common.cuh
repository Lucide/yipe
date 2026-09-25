#pragma once
#include <cuda/std/limits>
#include <cuda/std/mdspan>
#include <cuda_runtime.h>
#include "gpu-common.cuh"
#include "tables.cuh"

namespace marching_cubes {

using extents_3d =
    cuda::std::extents<u32, cuda::std::dynamic_extent, cuda::std::dynamic_extent, cuda::std::dynamic_extent>;

constexpr dim3 threads_per_block = {4, 4, 4};

struct Params {
  u32 points;
  float3 room_dim{};
  float particle_diameter;
  float particle_diameter_sq;
  uint3 cells{};
  uint3 fields{};
  float isolevel;

  Params(u32 const points, float3 const room_dim, float const particle_diameter, float const isolevel) {
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

// Estimate *unnormalised* Wendland C2 density
GPU float calculate_density(Point const node_pos, Point const point_pos, Params const params) {
  // NOTE: unnormalised C2 is fine for graphics but not for physics

  float const dist_sq = sq_dist(node_pos, point_pos);

  if (dist_sq >= params.particle_diameter_sq) {
    return 0.0f;
  }

  // fast sqrt: sqrt(x) == x * rsqrt(x). +epsilon guards against
  // rsqrtf(0) -> inf -> 0*inf -> NaN when node_pos == point_pos exactly.
  float const distance = dist_sq * rsqrtf(dist_sq + cuda::std::numeric_limits<float>::min());

  // Wendland C2 kernel
  float const u = distance / params.particle_diameter;
  float const term = 1.0f - u;
  float const w = (term * term * term * term) * (4.0f * u + 1.0f);
  return w;
}

GPU Point
interpolate_vertex(float const isolevel, Point const p1, Point const p2, float const field1, float const field2) {
  if (fabs(field1 - field2) <= 0.00001)
    return (p1 + p2) * 0.5f;
  return p1 + (p2 - p1) * (isolevel - field1) / (field2 - field1);
}

GPU Point inline get_cell_corner(uint3 const cell, u8 const corner, float const particle_diameter) {
  Offsets const offsets = get_cell_corner_offsets(corner);
  return {static_cast<float>(static_cast<double>(cell.x + offsets.x) * particle_diameter),
          static_cast<float>(static_cast<double>(cell.y + offsets.y) * particle_diameter),
          static_cast<float>(static_cast<double>(cell.z + offsets.z) * particle_diameter), 1};
}

GPU inline float select_field(u8 const corner, float const *const fields) {
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

GPU Point inline get_edge_vertex(
    u8 const edge, float const isolevel, uint3 const cell, float const particle_diameter, float const *const fields) {
  EdgeVertexes const vertexes = get_edge_vertexes(edge);

  Point const corner0 = get_cell_corner(cell, vertexes.c0, particle_diameter);
  Point const corner1 = get_cell_corner(cell, vertexes.c1, particle_diameter);

  float field0 = select_field(vertexes.c0, fields);

  float field1 = select_field(vertexes.c1, fields);

  return interpolate_vertex(isolevel, corner0, corner1, field0, field1);
}
} // namespace marching_cubes
