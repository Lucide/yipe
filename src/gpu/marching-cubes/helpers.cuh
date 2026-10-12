#pragma once
#include <cuda/std/limits>
#include <cuda/std/mdspan>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "tables.cuh"

namespace marching_cubes {

using extents_3d =
    cuda::std::extents<u32, cuda::std::dynamic_extent, cuda::std::dynamic_extent, cuda::std::dynamic_extent>;

inline constexpr dim3 threads_per_block = {4, 4, 4};

typedef struct SampleData {
  float field;
  float3 gradient;
} SampleData;

struct Params {
  u32 points;
  float3 room{};
  float particle_diameter;
  float particle_diameter_sq;
  float isolevel;
  u323 cells{};
  u323 samples{};

  Params(u32 const points, float3 const room_dim, float const particle_diameter, float const isolevel) {
    this->points = points;
    this->room = room_dim;
    this->particle_diameter = particle_diameter;
    this->particle_diameter_sq = particle_diameter * particle_diameter;
    this->isolevel = isolevel;
    this->cells = {static_cast<u32>(ceil(room_dim.x / particle_diameter)),
                   static_cast<u32>(ceil(room_dim.y / particle_diameter)),
                   static_cast<u32>(ceil(room_dim.z / particle_diameter))};
    this->samples = {this->cells.x + 1, this->cells.y + 1, this->cells.z + 1};
  }
  GPU HOST constexpr u32 cell_count() const { return this->cells.x * this->cells.y * this->cells.z; }
  GPU HOST constexpr u32 samples_count() const { return this->samples.x * this->samples.y * this->samples.z; }
};

GPU float3 get_grid_point(u323 const coords, float const particle_diameter) {
  return vec_cast<float3>(vec_cast<double3>(coords) * particle_diameter);
}

GPU bool is_out_of_bounds(u323 const coords, u323 const bounds) {
  return coords.x >= bounds.x || coords.y >= bounds.y || coords.z >= bounds.z;
}

GPU bool is_out_of_bounds(float3 const point, float3 const room) {
  return point.x <= 0 || point.y <= 0 || point.z <= 0 || point.x >= room.x || point.y >= room.y || point.z >= room.z;
}

GPU pair<s83, u83> get_neighbourhood_bounds(u323 const coords, u323 const samples) {
  s8 sx, sy, sz;
  u8 dx, dy, dz;
  if (coords.x == 0) {
    sx = 0;
    dx = 1;
  } else if (coords.x == samples.x - 1) {
    sx = -2;
    dx = 2;
  } else {
    sx = -1;
    dx = 2;
  }
  if (coords.y == 0) {
    sy = 0;
    dy = 1;
  } else if (coords.y == samples.y - 1) {
    sy = -2;
    dy = 2;
  } else {
    sy = -1;
    dy = 2;
  }
  if (coords.z == 0) {
    sz = 0;
    dz = 1;
  } else if (coords.z == samples.z - 1) {
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

template <typename T> GPU void add_checked_hash(T (&seen)[8], u8 &seen_count, T const hash) {
  switch (seen_count) {
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
    assert(seen_count < 8);
    return;
  }
  ++seen_count;
}

GPU inline SampleData get_corner_data(SampleData const (&data)[8], u8 const corner_index) {
  switch (corner_index) {
  case 0:
    return data[0];
  case 1:
    return data[1];
  case 2:
    return data[2];
  case 3:
    return data[3];
  case 4:
    return data[4];
  case 5:
    return data[5];
  case 6:
    return data[6];
  case 7:
    return data[7];
  default:
    assert(corner_index < 8);
    return {};
  }
}

// estimate *unnormalised* Wendland C2 density
// fine for graphics but not for physics
GPU float calculate_density(Params const params, float3 const sample_point, float3 const point) {
  float const dist_sq = sq_dist(sample_point, point);

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

GPU float3 interpolate(float3 const p1, float3 const p2, float const coefficient) {
  return (1 - coefficient) * p1 + coefficient * p2;
}

GPU pair<float3, float3>
get_edge_vertex(Params const &params, u8 const edge, u323 const cell, SampleData const (&data)[8]) {
  auto const [i1, i2] = get_edge_indices(edge);
  float3 const corner1 =
      clamp_point(get_grid_point(cell + get_cell_corner_offsets(i1), params.particle_diameter), params.room);
  float3 const corner2 =
      clamp_point(get_grid_point(cell + get_cell_corner_offsets(i2), params.particle_diameter), params.room);
  SampleData data1 = get_corner_data(data, i1);
  SampleData data2 = get_corner_data(data, i2);

  float const scale = fmaxf(fabsf(data1.field), fmaxf(fabsf(data2.field), fabsf(params.isolevel)));
  float const tolerance = 8.f * cuda::std::numeric_limits<float>::epsilon() * fmaxf(scale, 1.0f);
  float const coefficient = fabsf(data1.field - data2.field) <= tolerance
                                ? 0.5f
                                : (params.isolevel - data1.field) / (data2.field - data1.field);
  float3 const avg_gradient = interpolate(data1.gradient, data2.gradient, coefficient);
  float const sq_avg_gradient = avg_gradient * avg_gradient;
  return {interpolate(corner1, corner2, coefficient),
          ((sq_avg_gradient > 0.f) ? avg_gradient * -rsqrtf(sq_avg_gradient) : float3{})};
}

} // namespace marching_cubes
