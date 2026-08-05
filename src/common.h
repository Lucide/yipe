#ifndef YIPE_COMMON_H
#define YIPE_COMMON_H

#ifdef _MSC_VER
#pragma warning(disable : 5039)
#endif

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned int uint;
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

void launch_sort_points(GPUInfo const *info, Point *points, u32 size);

#ifdef __cplusplus
}
#endif

#endif
