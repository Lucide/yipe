#pragma once

#ifdef _MSC_VER
#pragma warning(disable : 5039)
#endif

#include <cuda_runtime_api.h>
#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

#define STR_HELPER(str) #str
#define XSTR(x) STR_HELPER(x)
#define CPRINT_HELPER(str) _Pragma(#str)
#define CPRINT(msg) CPRINT_HELPER(message("CPRINT: " msg))

#define ARRAY_LENGTH(arr) (sizeof(arr) / sizeof(*(arr)))
#define ARRAY_LENGTH_2D(arr) (sizeof(arr) / sizeof(**(arr)))
#define ARRAY_SIZE(ptr, length) ((length) * sizeof(*(ptr)))

// for API calls returning cudaError_t
#define TRAP_CUDAPI(call, msg_str)                                                                                     \
  do {                                                                                                                 \
    cudaError_t _err = (call);                                                                                         \
    handle_cuda_result(_err, (msg_str));                                                                               \
    handle_cuda_error_debug((msg_str));                                                                                \
  } while (0)

// for kernel launches
#define TRAP_KERNEL(call, msg_str)                                                                                     \
  do {                                                                                                                 \
    (call);                                                                                                            \
    handle_cuda_error_debug((msg_str));                                                                                \
  } while (0)

#define GPU __device__
#define HOST __host__
#define KERNEL __global__
#define SHARED __shared__
#define CONSTANT __constant__

typedef unsigned int uint;
typedef int8_t s8;
typedef uint8_t u8;
typedef int16_t s16;
typedef uint16_t u16;
typedef uint32_t u32;
typedef int32_t s32;
typedef uint64_t u64;
typedef int64_t s64;

typedef struct bool3 {
  bool x, y, z;
} bool3;

typedef struct u83 {
  u8 x, y, z;
} u83;

typedef struct s83 {
  s8 x, y, z;
} s83;

typedef struct u323 {
  u32 x, y, z;
} u323;

typedef struct s323 {
  s32 x, y, z;
} s323;

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

typedef struct float33 {
  float3 points[3];
} float33;

typedef struct Geometry {
  u32 count;
  u32 : 32;
  float33 *triangles;
  float33 *normals;
} Geometry;

typedef struct {
  u32 multiprocessors;
  u32 cores_per_multiprocessor;
} GPUInfo;

typedef struct HashTable HashTable;

void handle_cuda_error(char const *message);
void handle_cuda_result(cudaError_t err, char const *message);
void handle_cuda_error_debug(char const *message);

void launch_exclusive_scan(u32 const *src, u32 *dst, u32 length, cudaStream_t stream);
void launch_sort_points(GPUInfo const *info, float4 *points, u32 length, float particle_diameter, cudaStream_t stream);
struct HashTable *new_hashtable(
    GPUInfo const *info, float4 *points, u32 length, u32 buckets, float particle_diameter, cudaStream_t stream);
void launch_verify_hashtable(GPUInfo const *info,
                             float4 const *data,
                             u32 length,
                             HashTable const *table,
                             float particle_diameter,
                             cudaStream_t stream);
void launch_generate_points(GPUInfo const *info, float3 room, u32 seed, float4 *out, u32 length, cudaStream_t stream);
float *launch_compute_scalar_fields(GPUInfo const *info,
                                    float4 *data,
                                    u32 length,
                                    float3 room_dim,
                                    float particle_diameter,
                                    float isolevel,
                                    cudaStream_t stream);
void launch_preprocess_float3(GPUInfo const *info, float3 const *in, float4 *out, u32 length, cudaStream_t stream);
Geometry launch_marching_cubes(
    float const *fields, u32 length, float3 room_dim, float particle_diameter, float isolevel, cudaStream_t stream);
Geometry launch_triangulate(GPUInfo const *info,
                            float4 *data,
                            u32 length,
                            float3 room_dim,
                            float particle_diameter,
                            float isolevel,
                            cudaStream_t stream);
GPU HOST u32 point_hash(float4 point, u32 particle_count, float particle_diameter);

#ifdef __cplusplus
}
#endif
