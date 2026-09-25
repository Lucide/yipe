#pragma once

#ifdef _MSC_VER
#pragma warning(disable : 5039)
#endif

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#endif

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

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
typedef int8_t i8;
typedef uint8_t u8;
typedef int16_t s16;
typedef uint16_t u16;
typedef uint32_t u32;
typedef int32_t s32;
typedef uint64_t u64;

#define ROOM_WIDTH 600
#define ROOM_DEPTH 400
#define ROOM_HEIGHT 300
#define PARTICLE_DIAMETER 12.0f
#define ISOLEVEL 0.1f

static float3 const room_dims = {ROOM_WIDTH, ROOM_DEPTH, ROOM_HEIGHT};

static_assert(PARTICLE_DIAMETER > 0.0f, "PARTICLE_DIAMETER must be greater than 0");

typedef struct {
  u32 multiprocessors;
  u32 cores_per_multiprocessor;
} GPUInfo;

typedef struct rows4x4 {
  float4 r0, r1, r2, r3;
} rows4x4;

typedef struct float4x4 {
  float m00, m01, m02, m03;
  float m10, m11, m12, m13;
  float m20, m21, m22, m23;
  float m30, m31, m32, m33;
} float4x4;

typedef union mat4x4 {
  rows4x4 rows;
  float4x4 comp;
  float arr[16];
} mat4x4;

typedef float4 Point;

typedef struct Triangle {
  float3 points[3];
} Triangle;

typedef struct Geometry {
  u32 count;
  u32 : 32;
  Triangle *triangles;
} Geometry;

typedef struct HashTable HashTable;

void handle_cuda_error(char const *message);
void handle_cuda_result(cudaError_t err, char const *message);
void handle_cuda_error_debug(char const *message);

void launch_exclusive_scan(u32 const *src, u32 *dst, u32 length);
void launch_sort_points(GPUInfo const *info, Point *points, u32 length);
struct HashTable *new_hashtable(GPUInfo const *info, Point *points, u32 length, u32 buckets);
void launch_verify_hashtable(GPUInfo const *info, Point const *data, u32 length, HashTable const *table);
Geometry
triangulate(GPUInfo const *info, Point *data, u32 length, float3 room_dim, float particle_diameter, float isolevel);

GPU HOST u32 point_hash(Point point, u32 particle_count, float particle_diameter);

#ifdef __cplusplus
}
#endif
