#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "common.h"

#define ARR_LENGTH 300 // 2^31 / sizeof(Point) = 2^31 / 12 = 178956970

static u32 get_cores_per_SM(int major, int minor) {
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

static GPUInfo fetch_gpu_info(void) {
  int deviceID;
  cudaGetDevice(&deviceID);
  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceID);

  int const multiprocessors = props.multiProcessorCount;
  u32 const cores_per_multiprocessor = get_cores_per_SM(props.major, props.minor);

  return (GPUInfo){(u32)multiprocessors, cores_per_multiprocessor};
}

// Function to fill an array with random numbers within a specific range
static void generate_random_points(Point *const points, const u32 length, u32 seed) {
  srand(seed);
  for (u32 i = 0; i < length; ++i) {
    // (double)+ is the "unary plus sign operator" for double, it avoids a bad cast from u32 to double
    points[i] =
        (Point){.x = (float)((double)+rand() / (double)RAND_MAX * (double)room_dim.x),  // NOLINT(cert-msc50-cpp)
                .y = (float)((double)+rand() / (double)RAND_MAX * (double)room_dim.y),  // NOLINT(cert-msc50-cpp)
                .z = (float)((double)+rand() / (double)RAND_MAX * (double)room_dim.z)}; // NOLINT(cert-msc50-cpp)
  }
}

int main(int const argc, char const *const *const argv) {
  u32 const seed = 0;
  GPUInfo const info = fetch_gpu_info();

  u32 length = ARR_LENGTH;

  // rad size from argv if present
  if (argc > 1) {
    char *endptr;
    length = strtoul(argv[1], &endptr, 0);
    if (argv[1] == endptr) {
      printf("Invalid argument: \"%s\"", argv[1]);
      exit(1);
    }
  }

  printf("info:\n\tmultiprocessors: %u\n\tcores per multiprocessor: %u\n", info.multiprocessors,
         info.cores_per_multiprocessor);
  printf("seed %u\n", seed);

  struct Point *const cpu_points = (struct Point *)malloc(sizeof(Point) * length);
  generate_random_points(cpu_points, length, 549951);

  if (length <= 4096) {
    printf("\n printing unsorted points\n");
    for (u32 i = 0; i < length; ++i) {
      u32 hash1 = point_hash(cpu_points[i], length, PARTICLE_DIAMETER);
      printf("%u ", hash1);
    }
  }

  struct Point *gpu_points;
  handle_cuda_result(cudaMalloc((void **)&gpu_points, sizeof(Point) * length), "Main: Allocate points");
  handle_cuda_result(cudaMemcpy(gpu_points, cpu_points, sizeof(Point) * length, cudaMemcpyHostToDevice),
                     "Main: Load points");

  printf("\n");

  printf("launch\n");
  launch_sort_points(&info, gpu_points, length);
  printf("done\n");

  // Copy result back and clean up
  handle_cuda_result(cudaMemcpy(cpu_points, gpu_points, sizeof(Point) * length, cudaMemcpyDeviceToHost),
                     "Main: Unload points");

  if (length <= 4096) {
    printf("\n printing sorted points\n");
    for (u32 i = 0; i < length; ++i) {
      u32 hash1 = point_hash(cpu_points[i], length, PARTICLE_DIAMETER);
      printf("%u ", hash1);
    }
  }

  printf("\n");

  u32 count = 0;
  for (u32 i = 0; i < length - 1; ++i) {
    u32 hash1 = point_hash(cpu_points[i], length, PARTICLE_DIAMETER);
    u32 hash2 = point_hash(cpu_points[i + 1], length, PARTICLE_DIAMETER);
    if (hash1 > hash2) {
      printf("sorting error: %u %u, index: %u %u\n", hash1, hash2, i, i + 1);
      ++count;
    }
  }

  printf("errors: %u\n", count);

  HashTable *table = new_hashtable(&info, gpu_points, length);
  printf("hashtable done\n");
  verify_hashtable(&info, gpu_points, length, table);
  handle_cuda_error("collect `verify_hashtable` errors");
  printf("hashtable verified\n");

  launch_exclusive_scan(((u32 *)table), ((u32 *)table), length);
  handle_cuda_result(cudaFree(gpu_points), "Main: Free points");
  handle_cuda_result(cudaFree(table), "Main: Free table");
  printf("scan done\n");

  free(cpu_points);
  return 0;
}
