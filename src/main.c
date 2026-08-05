#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define ARR_SIZE 200000 // 2^31 / sizeof(Point) = 2^31 / 12 = 178956970
#define ROOM_WIDTH 600.
#define ROOM_DEPTH 400.
#define ROOM_HEIGHT 300.

int get_cores_per_SM(int major, int minor) {
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
void generate_random_points(Point *const points, uint seed) {
  // 1. Initialize the random number generator with the provided seed
  srand(seed);

  // 2. Loop through the array and assign a random number to each element
  for (int i = 0; i < ARR_SIZE; ++i) {
    points[i] = (Point){.x = (float)((double)rand() / RAND_MAX * ROOM_WIDTH),
                        .y = (float)((double)rand() / RAND_MAX * ROOM_DEPTH),
                        .z = (float)((double)rand() / RAND_MAX * ROOM_HEIGHT)};
  }
}

int main(int const argc, char const *const *const argv) {
  GPUInfo const info = fetch_gpu_info();

  struct Point *const cpu_points = malloc(sizeof(Point[ARR_SIZE]));
  generate_random_points(cpu_points, 0);

  struct Point *gpu_points;
  cudaMalloc((void **)&gpu_points, sizeof(Point[ARR_SIZE]));
  cudaMemcpy(gpu_points, cpu_points, sizeof(Point[ARR_SIZE]), cudaMemcpyHostToDevice);

  printf("Launch\n");
  launch_sort_points(&info, gpu_points, ARR_SIZE);

  // Copy result back and clean up
  cudaFree(gpu_points);

  return 0;
}
