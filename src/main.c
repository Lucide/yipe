#include <math.h>
#include <raylib.h>
#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "viewer/viewer.h"

#define ARR_LENGTH 10

static u32 get_cores_per_sm(int major, int minor) {
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
  u32 const cores_per_multiprocessor = get_cores_per_sm(props.major, props.minor);

  return (GPUInfo){(u32)multiprocessors, cores_per_multiprocessor};
}

// Function to fill an array with random numbers within a specific range
static void generate_random_points(Point *const points, const u32 length, u32 seed) {
  srand(seed);
  for (u32 i = 0; i < length; ++i) {
    // (double)+ is the "unary plus sign operator" for double, it avoids a bad cast from u32 to double
    points[i] =
        (Point){.x = (float)(((double)+rand() / (double)RAND_MAX) * (double)room_dims.x), // NOLINT(cert-msc50-cpp)
                .y = (float)(((double)+rand() / (double)RAND_MAX) * (double)room_dims.y), // NOLINT(cert-msc50-cpp)
                .z = (float)(((double)+rand() / (double)RAND_MAX) * (double)room_dims.z), // NOLINT(cert-msc50-cpp)
                .w = 1};
  }
}

// static void free_gpu_geometry(Geometry const geometry) {
//   handle_cuda_result(cudaFree(geometry.triangles), "Main: Free triangles");
// }

static Geometry download_geometry(Geometry const gpu_geometry) {
  Geometry cpu_geometry =
      (Geometry){.count = gpu_geometry.count, .triangles = malloc(gpu_geometry.count * sizeof(Triangle))};
  handle_cuda_result(cudaMemcpy(cpu_geometry.triangles, gpu_geometry.triangles, gpu_geometry.count * sizeof(Triangle),
                                cudaMemcpyDeviceToHost),
                     "Main: download triangles");
  handle_cuda_result(cudaFree(gpu_geometry.triangles), "Main: Free triangles");
  return cpu_geometry;
}

static void free_geometry(Geometry const geometry) { free(geometry.triangles); }

static void print_triangles(Geometry const geometry) {
  for (u32 i = 0; i < geometry.count; ++i) {
    const Triangle triangle = geometry.triangles[i];
    printf("(%.1f, %.1f, %.1f) (%.1f, %.1f, %.1f) (%.1f, %.1f, %.1f)\n", (double)triangle.points[0].x,
           (double)triangle.points[0].y, (double)triangle.points[0].z, (double)triangle.points[1].x,
           (double)triangle.points[1].y, (double)triangle.points[1].z, (double)triangle.points[2].x,
           (double)triangle.points[2].y, (double)triangle.points[2].z);
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
      printf("Invalid argument: \"%s\"\n", argv[1]);
      exit(1);
    }
  }

  printf("info:\n\tmultiprocessors: %u\n\tcores per multiprocessor: %u\n", info.multiprocessors,
         info.cores_per_multiprocessor);
  printf("seed %u\n", seed);

  Point *const cpu_points = (Point *)malloc(sizeof(Point) * length);
  generate_random_points(cpu_points, length, 549951);

  Point *gpu_points;
  handle_cuda_result(cudaMalloc((void **)&gpu_points, sizeof(Point) * length), "Main: Allocate points");
  handle_cuda_result(cudaMemcpy(gpu_points, cpu_points, sizeof(Point) * length, cudaMemcpyHostToDevice),
                     "Main: Load points");
  printf("points loaded\n");

  Viewer viewer = new_viewer();

  for (u16 fc = 0; !WindowShouldClose(); ++fc) {
    Geometry gpu_geometry = triangulate(&info, gpu_points, length, room_dims, PARTICLE_DIAMETER, ISOLEVEL);
    Geometry geometry = download_geometry(gpu_geometry);

    // if (fc == 0) {
    //   print_triangles(geometry);
    // }

    if (OWNERSHIP_BORROWED == render(&viewer, &geometry)) {
      free_geometry(geometry);
    }

    if (fc % 60 == 0) {
      char titleBuffer[256];
      snprintf(titleBuffer, sizeof(titleBuffer), "CUDA Dynamic Triangle Renderer | Triangles: %u | FPS: %d",
               gpu_geometry.count, GetFPS());
      SetWindowTitle(titleBuffer);
    }
  }

  CloseWindow();

  handle_cuda_result(cudaFree(gpu_points), "Main: Free points");
  free(cpu_points);
  return 0;
}

[[maybe_unused]] static void test(const u32 length) {
  u32 const seed = 0;
  GPUInfo const info = fetch_gpu_info();

  printf("info:\n\tmultiprocessors: %u\n\tcores per multiprocessor: %u\n", info.multiprocessors,
         info.cores_per_multiprocessor);
  printf("seed %u\n", seed);

  Point *const cpu_points = (Point *)malloc(sizeof(Point) * length);
  generate_random_points(cpu_points, length, 549951);

  if (length <= 4096) {
    printf("\n printing unsorted points\n");
    for (u32 i = 0; i < length; ++i) {
      u32 hash1 = point_hash(cpu_points[i], length, PARTICLE_DIAMETER);
      printf("%u ", hash1);
    }
  }

  Point *gpu_points;
  handle_cuda_result(cudaMalloc((void **)&gpu_points, sizeof(Point) * length), "Main: Allocate points");
  handle_cuda_result(cudaMemcpy(gpu_points, cpu_points, sizeof(Point) * length, cudaMemcpyHostToDevice),
                     "Main: Load points");
  printf("points loaded\n");

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

  HashTable *table = new_hashtable(&info, gpu_points, length, length);
  u32 *cpu_table = (u32 *)malloc(sizeof(u32) * length + 1 + sizeof(u32));
  cudaMemcpy(cpu_table, table, sizeof(float) * length + 1 + sizeof(u32), cudaMemcpyDeviceToHost);
  if (length <= 4096) {
    for (u32 i = 0; i < length - 1; ++i) {
      u32 b = cpu_table[1 + i];
      u32 e = cpu_table[1 + i + 1];
      if (b != 0 && e != 0 && b != e) {
        printf("cell %u: %u, %u\n", i, b - 1, e - 1);
      }
    }
  }
  printf("hashtable done\n");
  launch_verify_hashtable(&info, gpu_points, length, table);
  handle_cuda_error("collect `verify_hashtable` errors");
  printf("hashtable verified\n");

  launch_exclusive_scan(((const u32 *)table), ((u32 *)table), length);
  handle_cuda_result(cudaFree(table), "Main: Free table");
  printf("scan done\n");

  Geometry geometry = triangulate(&info, gpu_points, length, room_dims, PARTICLE_DIAMETER, ISOLEVEL);
  if (geometry.count <= 1024) {
    Triangle *const triangles = (Triangle *)malloc(sizeof(Triangle) * geometry.count);
    cudaMemcpy(triangles, geometry.triangles, sizeof(Triangle) * geometry.count, cudaMemcpyDeviceToHost);
    for (u32 i = 0; i < geometry.count - 1; ++i) {
      const Triangle triangle = triangles[i];
      printf("(%.1f, %.1f, %.1f) (%.1f, %.1f, %.1f) (%.1f, %.1f, %.1f)\n", (double)triangle.points[0].x,
             (double)triangle.points[0].y, (double)triangle.points[0].z, (double)triangle.points[1].x,
             (double)triangle.points[1].y, (double)triangle.points[1].z, (double)triangle.points[2].x,
             (double)triangle.points[2].y, (double)triangle.points[2].z);
    }
    free(triangles);
  }

  handle_cuda_result(cudaFree(gpu_points), "Main: Free points");
  handle_cuda_result(cudaFree(geometry.triangles), "Main: Free triangles");
  printf("triangulation done\n");

  free(cpu_points);
}
