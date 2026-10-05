#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

static u32 get_cores_per_sm(int const major, int const minor) {
  switch (major) {
  case 2: // Fermi
    return (minor == 1) ? 48 : 32;
  case 3: // Kepler
    return 192;
  case 5: // Maxwell
    return 128;
  case 6: // Pascal
    if (minor == 0)
      return 64;
    if (minor == 1 || minor == 2)
      return 128;
    break;
  case 7: // Volta & Turing
    if (minor == 0 || minor == 5)
      return 64;
    break;
  case 8: // Ampere & Hopper
    if (minor == 0)
      return 64; // GA100 / H100
    if (minor == 6 || minor == 9)
      return 128; // GA10x / Ada Lovelace
    break;
  case 9: // Blackwell
    if (minor == 0)
      return 128;
    break;
  default:
    break;
  }
  return 128; // Default fallback for newer unknown architectures
}

GPUInfo fetch_gpu_info(void) {
  int deviceID;
  cudaGetDevice(&deviceID);
  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceID);

  int const multiprocessors = props.multiProcessorCount;
  u32 const cores_per_multiprocessor = get_cores_per_sm(props.major, props.minor);

  return (GPUInfo){(u32)multiprocessors, cores_per_multiprocessor};
}

// function to fill an array with random numbers within a specific range
void generate_random_points(float4 *const points, u32 const length, u32 const seed, float3 const room) {
  srand(seed);
  for (u32 i = 0; i < length; ++i) {
    // (double)+ is the "unary plus sign operator" for double, it avoids a bad cast from u32 to double
    points[i] = (float4){.x = (float)(((double)+rand() / (double)RAND_MAX) * (double)room.x), // NOLINT(cert-msc50-cpp)
                         .y = (float)(((double)+rand() / (double)RAND_MAX) * (double)room.y), // NOLINT(cert-msc50-cpp)
                         .z = (float)(((double)+rand() / (double)RAND_MAX) * (double)room.z), // NOLINT(cert-msc50-cpp)
                         .w = 1};
  }
}

void print_triangles(Geometry const geometry) {
  for (u32 i = 0; i < geometry.count; ++i) {
    float33 const triangle = geometry.triangles[i];
    printf("(%.1f, %.1f, %.1f) (%.1f, %.1f, %.1f) (%.1f, %.1f, %.1f)\n", (double)triangle.points[0].x,
           (double)triangle.points[0].y, (double)triangle.points[0].z, (double)triangle.points[1].x,
           (double)triangle.points[1].y, (double)triangle.points[1].z, (double)triangle.points[2].x,
           (double)triangle.points[2].y, (double)triangle.points[2].z);
  }
}

void free_gpu_geometry(Geometry const geometry) {
  handle_cuda_result(cudaFree(geometry.triangles), "main: free triangles");
}

Geometry download_geometry(Geometry const gpu_geometry, cudaStream_t const stream) {
  Geometry cpu_geometry =
      (Geometry){.count = gpu_geometry.count, .triangles = malloc(2 * gpu_geometry.count * sizeof(float33))};
  handle_cuda_result(cudaMemcpyAsync(cpu_geometry.triangles, gpu_geometry.triangles,
                                     2 * gpu_geometry.count * sizeof(float33), cudaMemcpyDeviceToHost, stream),
                     "main: download triangles");
  handle_cuda_result(cudaStreamSynchronize(stream), "main: download triangles");
  handle_cuda_result(cudaFree(gpu_geometry.triangles), "main: free triangles");
  cpu_geometry.normals = cpu_geometry.triangles + cpu_geometry.count;
  return cpu_geometry;
}

void free_geometry(Geometry const geometry) { free(geometry.triangles); }
