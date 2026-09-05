#pragma once

#include "gpu-common.cuh"

namespace marching_cubes {

CONSTANT u8 triangle_count[256];
CONSTANT Triangle triangles[256][5];

struct Sizes {
  u32 points;
  float3 room_dim{};
  float particle_diameter;
  uint3 cells{};
  uint3 fields{};

  Sizes(const u32 points, const float3 room_dim, const float particle_diameter) {
    this->points = points;
    this->room_dim = room_dim;
    this->particle_diameter = particle_diameter;
    this->cells = {static_cast<u32>(ceil(room_dim.x / particle_diameter)),
                   static_cast<u32>(ceil(room_dim.y / particle_diameter)),
                   static_cast<u32>(ceil(room_dim.z / particle_diameter))};
    this->fields = {this->cells.x + 1, this->cells.y + 1, this->cells.z + 1};
  }

  u32 fields_count() const { return this->fields.x * this->fields.y * this->fields.z; }
};

KERNEL void
compute_scalar_fields(const Point *const points, const HashTable *const table, const Sizes sizes, float *const fields) {
}
} // namespace marching_cubes
