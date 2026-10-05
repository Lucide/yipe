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

inline constexpr dim3 threads_per_block = {4, 4, 4};

struct Params {
  u32 points;
  float3 room{};
  float particle_diameter;
  float particle_diameter_sq;
  u323 cells{};
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
    this->fields = {this->cells.x + 1, this->cells.y + 1, this->cells.z + 1};
  }
  GPU HOST constexpr u32 cell_count() const { return this->cells.x * this->cells.y * this->cells.z; }
  GPU HOST constexpr u32 field_count() const { return this->fields.x * this->fields.y * this->fields.z; }
};

GPU bool is_out_of_bounds(u323 const coords, u323 const bounds) {
  return coords.x >= bounds.x || coords.y >= bounds.y || coords.z >= bounds.z;
}

GPU bool is_out_of_bounds(float3 const point, float3 const room) {
  return point.x <= 0 || point.y <= 0 || point.z <= 0 || point.x >= room.x || point.y >= room.y || point.z >= room.z;
}

GPU float3 clamp_point(float3 const point, float3 const room) {
  // assume positive coordinates
  return {fmin(point.x, room.x), fmin(point.y, room.y), fmin(point.z, room.z)};
}

GPU pair<s83, u83> get_neighbourhood_bounds(u323 const coords, u323 const fields) {
  s8 sx, sy, sz;
  u8 dx, dy, dz;
  if (coords.x == 0) {
    sx = 0;
    dx = 1;
  } else if (coords.x == fields.x - 1) {
    sx = -2;
    dx = 2;
  } else {
    sx = -1;
    dx = 2;
  }
  if (coords.y == 0) {
    sy = 0;
    dy = 1;
  } else if (coords.y == fields.y - 1) {
    sy = -2;
    dy = 2;
  } else {
    sy = -1;
    dy = 2;
  }
  if (coords.z == 0) {
    sz = 0;
    dz = 1;
  } else if (coords.z == fields.z - 1) {
    sz = -2;
    dz = 2;
  } else {
    sz = -1;
    dz = 2;
  }
  return {
      {sx, sy, sz},
      {dx, dy, dz}
  };
}

template <typename T> GPU void add_checked_hash(T (&seen)[8], u8 &checked_count, T const hash) {
  switch (checked_count) {
  case 0:
    seen[0] = hash;
    break;
  case 1:
    seen[1] = hash;
    break;
  case 2:
    seen[2] = hash;
    break;
  case 3:
    seen[3] = hash;
    break;
  case 4:
    seen[4] = hash;
    break;
  case 5:
    seen[5] = hash;
    break;
  case 6:
    seen[6] = hash;
    break;
  case 7:
    seen[7] = hash;
    break;
  default:
    assert(checked_count < 8);
    return;
  }
  ++checked_count;
}

// estimate *unnormalised* Wendland C2 density
// fine for graphics but not for physics
GPU float calculate_density(float3 const node_pos, float3 const point_pos, Params const params) {
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

GPU float3 get_grid_point(u323 const coords, float const particle_diameter) {
  return vec_cast<float3>(vec_cast<double3>(coords) * particle_diameter);
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

GPU float3
interpolate_vertex(float const isolevel, float3 const p1, float3 const p2, float const field1, float const field2) {
  float const scale = fmaxf(fabsf(field1), fmaxf(fabsf(field2), fabsf(isolevel)));
  float const tolerance = 8.f * cuda::std::numeric_limits<float>::epsilon() * fmaxf(scale, 1.0f);

  if (fabsf(field1 - field2) <= tolerance)
    // numerical stability
    return (p1 + p2) * 0.5f;
  return p1 + (p2 - p1) * (isolevel - field1) / (field2 - field1);
}

GPU float3 get_edge_vertex(Params const &params, u8 const edge, u323 const cell, float const (&fields)[8]) {
  EdgeVertexes const vertexes = get_edge_vertexes(edge);

  float field0 = select_field(vertexes.c0, fields);
  float field1 = select_field(vertexes.c1, fields);
  float3 const corner0 = get_grid_point(cell + get_cell_corner_offsets(vertexes.c0), params.particle_diameter);
  float3 const corner1 = get_grid_point(cell + get_cell_corner_offsets(vertexes.c1), params.particle_diameter);

  return interpolate_vertex(params.isolevel, clamp_point(corner0, params.room), clamp_point(corner1, params.room),
                            field0, field1);
}

} // namespace marching_cubes
