#pragma once

#ifdef _MSC_VER
#pragma warning(disable : 5039)
#endif

#ifdef __cplusplus
#define GPU_HOST __device__ __host__
#include <cstdint>
extern "C" {
#else
#define GPU_HOST
#include <stdint.h>
#endif

#define ARRAY_LENGTH(arr) (sizeof(arr) / sizeof(*(arr)))
#define ARRAY_LENGTH_2D(arr) (sizeof(arr) / sizeof(**(arr)))

#define gpu __device__
#define host __host__
#define kernel __global__
#define shared __shared__

typedef unsigned int uint;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;

#define PARTICLE_DIAMETER 1.0f
static_assert(PARTICLE_DIAMETER > 0.0f, "PARTICLE_DIAMETER must be greater than 0");

typedef struct {
  u32 multiprocessors;
  u32 cores_per_multiprocessor;
} GPUInfo;

typedef struct Point {
  float x, y, z;
} Point;

void launch_sort_points(GPUInfo const *info, Point *points, u32 length);

gpu host u32 point_hash(const Point *point, u32 particle_count);

#ifdef __cplusplus
}
#endif
