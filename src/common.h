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

#define STR_HELPER(str) #str
#define XSTR(x) STR_HELPER(x)
#define CPRINT_HELPER(str) _Pragma(#str)
#define CPRINT(msg) CPRINT_HELPER(message(msg))

#define ARRAY_LENGTH(arr) (sizeof(arr) / sizeof(*(arr)))
#define ARRAY_LENGTH_2D(arr) (sizeof(arr) / sizeof(**(arr)))

#define GPU __device__
#define HOST __host__
#define KERNEL __global__
#define SHARED __shared__

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

typedef struct HashTable HashTable;

void handle_cuda_error(const char *message);
void handle_cuda_result(cudaError_t err, const char *message);

void launch_exclusive_scan(u32 *data, u32 length);
void launch_sort_points(GPUInfo const *info, Point *points, u32 length);
struct HashTable *new_hashtable(GPUInfo const *info, Point *points, u32 length);

GPU HOST u32 point_hash(const Point *point, u32 particle_count);

#ifdef __cplusplus
}
#endif
