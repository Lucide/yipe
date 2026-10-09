#include "common.h"

bool read32_file(char const *filepath, char const **out_buffer, u32 *out_size);
GPUInfo fetch_gpu_info(void);
void generate_random_points(float3 *points, u32 length, u32 seed, float3 room);
void print_triangles(Geometry geometry);
void free_gpu_geometry(Geometry geometry);
Geometry download_geometry(Geometry gpu_geometry, cudaStream_t stream);
void free_geometry(Geometry geometry);
