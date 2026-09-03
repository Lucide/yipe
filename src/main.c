#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "common.h"

#define ARR_SIZE 550 // 2^31 / sizeof(Point) = 2^31 / 12 = 178956970
#define ROOM_WIDTH 600.
#define ROOM_DEPTH 400.
#define ROOM_HEIGHT 300.

int get_cores_per_SM(u32 major, u32 minor) {
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

GPUInfo fetch_gpu_info() {
  int deviceID;
  cudaGetDevice(&deviceID);
  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceID);

  int const multiprocessors = props.multiProcessorCount;
  int const cores_per_multiprocessor = get_cores_per_SM(props.major, props.minor);

  return (GPUInfo){multiprocessors, cores_per_multiprocessor};
}

// Function to fill an array with random numbers within a specific range
void generate_random_points(Point *const points, u32 seed) {
  // 1. Initialize the random number generator with the provided seed
  srand(seed);

  // 2. Loop through the array and assign a random number to each element
  for (int i = 0; i < ARR_SIZE; ++i) {
    points[i] = (Point){.x = (float)((double)rand() / RAND_MAX * ROOM_WIDTH),   // NOLINT(cert-msc50-cpp)
                        .y = (float)((double)rand() / RAND_MAX * ROOM_DEPTH),   // NOLINT(cert-msc50-cpp)
                        .z = (float)((double)rand() / RAND_MAX * ROOM_HEIGHT)}; // NOLINT(cert-msc50-cpp)
  }
}

int main(int const argc, char const *const *const argv) {
  u32 const seed = 0;
  GPUInfo const info = fetch_gpu_info();

  printf("info:\n\tmultiprocessors: %d\n\tcores per multiprocessor: %d\n", info.multiprocessors,
         info.cores_per_multiprocessor);
  printf("seed %d\n", seed);

  struct Point *const cpu_points = malloc(sizeof(Point[ARR_SIZE]));
  generate_random_points(cpu_points, 549951);

  struct Point *gpu_points;
  handle_cuda_result(cudaMalloc((void **)&gpu_points, sizeof(Point[ARR_SIZE])), "Main: Allocate points");
  handle_cuda_result(cudaMemcpy(gpu_points, cpu_points, sizeof(Point[ARR_SIZE]), cudaMemcpyHostToDevice),
                     "Main: Load points");

  printf("\n printing unsorted points\n");
  for (u32 i = 0; i < ARR_SIZE && ARR_SIZE <= 4096; ++i) {
    u32 hash1 = point_hash(&cpu_points[i], ARR_SIZE);
    printf("%d ", hash1);
  }

  printf("\n");

  printf("launch\n");
  launch_sort_points(&info, gpu_points, ARR_SIZE);
  printf("done\n");

  // Copy result back and clean up

  handle_cuda_result(cudaMemcpy(cpu_points, gpu_points, sizeof(Point[ARR_SIZE]), cudaMemcpyDeviceToHost),
                     "Main: Unload points");
  cudaDeviceSynchronize();

  printf("\n printing sorted points\n");
  for (u32 i = 0; i < ARR_SIZE && ARR_SIZE <= 4096; ++i) {
    u32 hash1 = point_hash(&cpu_points[i], ARR_SIZE);
    printf("%d ", hash1);
  }

  printf("\n");

  u32 count = 0;
  for (u32 i = 0; i < ARR_SIZE - 1; ++i) {
    u32 hash1 = point_hash(&cpu_points[i], ARR_SIZE);
    u32 hash2 = point_hash(&cpu_points[i + 1], ARR_SIZE);
    if (hash1 > hash2) {
      printf("sorting error: %d %d, index: %d %d\n", hash1, hash2, i, i + 1);
      ++count;
    }
  }

  printf("errors: %d\n", count);

  // HashTable *table = new_hashtable(&info, gpu_points, ARR_SIZE);
  handle_cuda_result(cudaFree(gpu_points), "Main: Free points");
  printf("hashtable done\n");

  // launch_exclusive_scan(&info, ((u32 *)table), ARR_SIZE);
  // handle_cuda_result(cudaFree(table), "Main: Free table");
  printf("scan done\n");

  free(cpu_points);
  return 0;
}
