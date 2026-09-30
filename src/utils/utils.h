#include "common.h"

GPUInfo fetch_gpu_info(void);
void generate_random_points(float4 *points, u32 length, u32 seed, float3 room);
void print_triangles(Geometry geometry);
void free_gpu_geometry(Geometry geometry);
Geometry download_geometry(Geometry gpu_geometry);
void free_geometry(Geometry geometry);
