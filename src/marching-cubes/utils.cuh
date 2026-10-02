#pragma once
#include <cuda/std/limits>
#include <cuda/std/mdspan>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "gpu-common.cuh"
#include "tables.cuh"

namespace marching_cubes {

using extents_3d =
    cuda::std::extents<u32, cuda::std::dynamic_extent, cuda::std::dynamic_extent, cuda::std::dynamic_extent>;

constexpr dim3 threads_per_block = {4, 4, 4};

struct Params {
  u32 points;
  float3 room{};
  float particle_diameter;
  float particle_diameter_sq;
  u323 cells{};
  u323 outer_cells{};
  u323 fields{};
  float isolevel;

  Params(u32 const points, float3 const room_dim, float const particle_diameter, float const isolevel) {
    this->points = points;
    this->room = room_dim;
    this->particle_diameter = particle_diameter;
    this->particle_diameter_sq = particle_diameter * particle_diameter;
    this->isolevel = isolevel;
    this->cells = {static_cast<u32>(ceil(room_dim.x / particle_diameter)),
                   static_cast<u32>(ceil(room_dim.y / particle_diameter)),
                   static_cast<u32>(ceil(room_dim.z / particle_diameter))};
    // outer cells include an extra cell to the left and one to the right in every dimension
    this->outer_cells = {this->cells.x + 2, this->cells.y + 2, this->cells.z + 2};
    this->fields = {this->cells.x + 1, this->cells.y + 1, this->cells.z + 1};
  }
  GPU HOST constexpr u32 cell_count() const { return this->cells.x * this->cells.y * this->cells.z; }
  GPU HOST constexpr u32 outer_cell_count() const {
    return this->outer_cells.x * this->outer_cells.y * this->outer_cells.z;
  }
  GPU HOST constexpr u32 field_count() const { return this->fields.x * this->fields.y * this->fields.z; }
};

// estimate *unnormalised* Wendland C2 density
// fine for graphics but not for physics
GPU float calculate_density(float4 const node_pos, float4 const point_pos, Params const params) {
  float const dist_sq = sq_dist(node_pos, point_pos);

  if (dist_sq >= params.particle_diameter_sq) {
    return 0.0f;
  }

  // fast sqrt: sqrt(x) == x * rsqrt(x). +epsilon guards against
  // rsqrtf(0) -> inf -> 0*inf -> NaN when node_pos == point_pos exactly
  float const distance = dist_sq * rsqrtf(dist_sq + cuda::std::numeric_limits<float>::min());

  // Wendland C2 kernel
  float const u = distance / params.particle_diameter;
  float const term = 1.0f - u;
  float const w = (term * term * term * term) * (4.0f * u + 1.0f);
  return w;
}

GPU bool is_out_of_bounds(float4 const point, float3 const room) {
  return point.x <= 0 || point.y <= 0 || point.z <= 0 || point.x >= room.x || point.y >= room.y || point.z >= room.z;
}

GPU float4
interpolate_vertex(float const isolevel, float4 const p1, float4 const p2, float const field1, float const field2) {
  if (field1 != isolevel && field2 != isolevel && fabsf(field1 - field2) <= 0.00001)
    // numerical stability
    return (p1 + p2) * 0.5f;
  return p1 + (p2 - p1) * (isolevel - field1) / (field2 - field1);
}

GPU float4 get_cell_corner(s323 const cell, u8 const corner, float const particle_diameter) {
  Offsets const offsets = get_cell_corner_offsets(corner);
  return {dtof(static_cast<double>(cell.x + offsets.x) * particle_diameter),
          dtof(static_cast<double>(cell.y + offsets.y) * particle_diameter),
          dtof(static_cast<double>(cell.z + offsets.z) * particle_diameter), 1};
}

GPU inline float select_field(u8 const corner, float const (&fields)[8]) {
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

GPU float4 get_edge_vertex(Params const &params, u8 const edge, s323 const cell, float const (&fields)[8]) {
  EdgeVertexes const vertexes = get_edge_vertexes(edge);

  float4 const corner0 = get_cell_corner(cell, vertexes.c0, params.particle_diameter);
  float4 const corner1 = get_cell_corner(cell, vertexes.c1, params.particle_diameter);

  float field0 = select_field(vertexes.c0, fields);
  float field1 = select_field(vertexes.c1, fields);

  if (is_out_of_bounds(corner0, params.room) || is_out_of_bounds(corner1, params.room)) {
    // branchless with selp instruction
    // 0.f is hardcoded value
    field0 = field0 > 0.f ? params.isolevel : 0.0f;
    field1 = field1 > 0.f ? params.isolevel : 0.0f;
  }

  return interpolate_vertex(params.isolevel, corner0, corner1, field0, field1);
}

GPU float3 clamp_point(float3 const point, float3 const room) {
  return {fmin(fmax(point.x, 0.f), room.x), fmin(fmax(point.y, 0.f), room.y), fmin(fmax(point.z, 0.f), room.z)};
}
} // namespace marching_cubes
