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
#define CONSTANT __constant__

typedef unsigned int uint;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

#define ROOM_WIDTH 600
#define ROOM_DEPTH 400
#define ROOM_HEIGHT 300
#define PARTICLE_DIAMETER 30.0f

static const float3 room_dim = {ROOM_WIDTH, ROOM_DEPTH, ROOM_HEIGHT};

static_assert(PARTICLE_DIAMETER > 0.0f, "PARTICLE_DIAMETER must be greater than 0");

typedef struct {
  u32 multiprocessors;
  u32 cores_per_multiprocessor;
} GPUInfo;

typedef struct Point {
  float x, y, z;
} Point;

typedef struct Triangle {
  Point points[3];
} Triangle;

typedef struct HashTable HashTable;

void handle_cuda_error(const char *message);
void handle_cuda_result(cudaError_t err, const char *message);
void handle_cuda_error_debug(const char *message);

void launch_exclusive_scan(const u32 *src, u32 *dst, u32 length);
void launch_sort_points(GPUInfo const *info, Point *points, u32 length);
struct HashTable *new_hashtable(GPUInfo const *info, Point *points, u32 length);
void verify_hashtable(GPUInfo const *info, const Point *data, u32 length, const HashTable *table);

GPU HOST u32 point_hash(Point point, u32 particle_count, float particle_diameter);

#ifdef __cplusplus
}
#endif
