#include <argp.h>
#include <driver_types.h>
#include <math.h>
#include <raylib.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>
#include "common.h"
#include "streamer/streamer.h"
#include "utils/utils.h"
#include "viewer/viewer.h"

static char const doc[] = "yipe -- A onesweep/spatial-hashing/marching-cubes demo.";
static char const args_doc[] = "";

static struct argp_option const options[] = {
    /* { name, key, arg, flags, doc, group } */
    {             "seed", 's', "NUM", 0,                 "Seed for random points", 0},
    {   "particle-count", 'c', "NUM", 0,                    "Number of particles", 0},
    {"particle-diameter", 'd', "NUM", 0,              "Diameter of the particles", 0},
    {         "isolevel", 'i', "NUM", 0, "Isolevel for generating the isosurface", 0},
    {       "room-width", 'x', "NUM", 0,                   "Room width in metres", 0},
    {      "room-height", 'y', "NUM", 0,                  "Room height in metres", 0},
    {       "room-depth", 'z', "NUM", 0,                   "Room depth in metres", 0},
    {             "test", 't',  NULL, 0,         "Run test instead of visualizer", 0},
    {               NULL,   0,  NULL, 0,                                     NULL, 0}  /* terminating entry */
};

typedef struct Arguments {
  u32 seed;
  u32 particle_count;
  float particle_diameter;
  float3 room;
  float isolevel;
  bool test;
} Arguments;

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
  Arguments *arguments = state->input;

  char *endptr;
  switch (key) {
  case 's':
    u32 const seed = strtoul(arg, &endptr, 0);
    if (arg == endptr) {
      argp_failure(state, 1, 0, TextFormat("Invalid argument: \"%s\"\n; seed must be an integer number", arg));
    }
    arguments->seed = seed;
    break;
  case 'c':
    u32 const particle_count = strtoul(arg, &endptr, 0);
    if (arg == endptr) {
      argp_failure(state, 1, 0, "Invalid argument: \"%s\"\n; particle count must be an integer number", arg);
    }
    arguments->particle_count = particle_count;
    break;
  case 'd':
    float const particle_diameter = strtof(arg, &endptr);
    if (arg == endptr) {
      argp_failure(state, 1, 0, TextFormat("Invalid argument: \"%s\"\n; particle diameter must be a number", arg));
    }
    arguments->particle_diameter = particle_diameter;
    break;
  case 'i':
    float const isolevel = strtof(arg, &endptr);
    if (arg == endptr) {
      argp_failure(state, 1, 0, "Invalid argument: \"%s\"\n; isolevel must be a number", arg);
    }
    arguments->isolevel = isolevel;
    break;
  case 'x':
    float const x = strtof(arg, &endptr);
    if (arg == endptr) {
      argp_failure(state, 1, 0, "Invalid argument: \"%s\"\n; room width must be a number", arg);
    }
    arguments->room.x = x;
    break;
  case 'y':
    float const y = strtof(arg, &endptr);
    if (arg == endptr) {
      argp_failure(state, 1, 0, "Invalid argument: \"%s\"\n; room height must be a number", arg);
    }
    arguments->room.y = y;
    break;
  case 'z':
    float const z = strtof(arg, &endptr);
    if (arg == endptr) {
      argp_failure(state, 1, 0, "Invalid argument: \"%s\"\n; room depth must be a number", arg);
    }
    arguments->room.z = z;
    break;
  case 't':
    arguments->test = true;
    break;
  case ARGP_KEY_ARG:
    argp_failure(state, 1, 0, "Unexprected positional argument");
    break;
  case ARGP_KEY_END:
    /* Validation logic when argument parsing completes */
    if (arguments->isolevel <= 0) {
      argp_failure(state, 1, 0, "Isolevel must be greater than 0");
    }
    if (arguments->particle_diameter <= 0) {
      argp_failure(state, 1, 0, "Particle diameter must be greater than 0");
    }
    if (arguments->room.x <= 0 || arguments->room.y <= 0 || arguments->room.z <= 0) {
      argp_failure(state, 1, 0, "All room dimensions must be greater than 0");
    }
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp const argp = {options, parse_opt, args_doc, doc, NULL, NULL, NULL};

static int player(Arguments args, GPUInfo info);
static void random_points(Arguments args, GPUInfo info);
static int test(Arguments args, GPUInfo info);

int main(int const argc, char **const argv) {
  Arguments args = (Arguments){
      .seed = 549951,
      .particle_count = 200,
      .particle_diameter = .06f,
      .room = {6.f, 3.f, 4.f},
      .isolevel = 0.1f,
      .test = false,
  };
  argp_parse(&argp, argc, argv, 0, NULL, &args);
  GPUInfo const info = fetch_gpu_info();
  printf("info:\n\tmultiprocessors: %u\n\tcores per multiprocessor: %u\n", info.multiprocessors,
         info.cores_per_multiprocessor);

  // TODO

  player(args, info);
  // if (args.test) {
  //   test(args, info);
  //   return 0;
  // }

  // random_points(args, info);
  return 0;
}

static int player(Arguments args, GPUInfo const info) {
  AsyncFluidStreamer *streamer =
      streamer_create("D:/Users/Nemo/OneDrive-shared/cuda/yipe/pysrc/res/fluid/DoubleDamBreakWithSphere.bin");
  if (!streamer)
    return -1;

  int active_buf = 0;

  Viewer viewer = new_viewer(args.room, &args.particle_diameter, &args.isolevel);

  while (!WindowShouldClose()) {
    float4 *gpu_points = streamer_acquire_next_frame(streamer, active_buf, cudaStreamPerThread);

    Geometry gpu_geometry = triangulate(&info, gpu_points, (u32)streamer->num_particles, args.room,
                                        args.particle_diameter, args.isolevel, cudaStreamPerThread);
    present(&viewer, gpu_geometry, cudaStreamPerThread);
    streamer_release_frame(streamer, active_buf, cudaStreamPerThread);
    active_buf = 1 - active_buf;
    set_window_title("CUDA Marching Cubes | Triangles: %u | FPS: %d", gpu_geometry.count, GetFPS());
  }

  unload_viewer(&viewer);
  CloseWindow();
  streamer_destroy(streamer);
  return 0;
}

static void random_points(Arguments args, GPUInfo const info) {
  printf("seed %u\n", args.seed);

  float4 *const cpu_points = (float4 *)malloc(sizeof(float4) * args.particle_count);
  generate_random_points(cpu_points, args.particle_count, args.seed, args.room);

  float4 *gpu_points;
  handle_cuda_result(cudaMalloc((void **)&gpu_points, sizeof(float4) * args.particle_count), "Main: Allocate points");
  handle_cuda_result(cudaMemcpy(gpu_points, cpu_points, sizeof(float4) * args.particle_count, cudaMemcpyHostToDevice),
                     "Main: Load points");
  printf("points loaded\n");

  Viewer viewer = new_viewer(args.room, &args.particle_diameter, &args.isolevel);

  while (!WindowShouldClose()) {
    Geometry gpu_geometry = triangulate(&info, gpu_points, args.particle_count, args.room, args.particle_diameter,
                                        args.isolevel, cudaStreamPerThread);
    present(&viewer, gpu_geometry, cudaStreamPerThread);
    set_window_title("CUDA Marching Cubes | Triangles: %u | FPS: %d", gpu_geometry.count, GetFPS());
  }

  unload_viewer(&viewer);
  CloseWindow();

  handle_cuda_result(cudaFree(gpu_points), "Main: Free points");
  free(cpu_points);
}

static int test(Arguments const args, GPUInfo const info) {
  printf("seed %u\n", args.seed);

  float4 *const cpu_points = (float4 *)malloc(sizeof(float4) * args.particle_count);
  generate_random_points(cpu_points, args.particle_count, args.seed, args.room);

  if (args.particle_count <= 4096) {
    printf("\n printing unsorted points\n");
    for (u32 i = 0; i < args.particle_count; ++i) {
      u32 hash1 = point_hash(cpu_points[i], args.particle_count, args.particle_diameter);
      printf("%u ", hash1);
    }
  }

  float4 *gpu_points;
  handle_cuda_result(cudaMalloc((void **)&gpu_points, sizeof(float4) * args.particle_count), "Main: Allocate points");
  handle_cuda_result(cudaMemcpy(gpu_points, cpu_points, sizeof(float4) * args.particle_count, cudaMemcpyHostToDevice),
                     "Main: Load points");
  printf("points loaded\n");

  printf("\n");

  printf("sort points\n");
  launch_sort_points(&info, gpu_points, args.particle_count, args.particle_diameter, cudaStreamPerThread);
  printf("done\n");

  // Copy result back and clean up
  handle_cuda_result(cudaMemcpy(cpu_points, gpu_points, sizeof(float4) * args.particle_count, cudaMemcpyDeviceToHost),
                     "Main: Unload points");

  if (args.particle_count <= 4096) {
    printf("\n printing sorted points\n");
    for (u32 i = 0; i < args.particle_count; ++i) {
      u32 hash1 = point_hash(cpu_points[i], args.particle_count, args.particle_diameter);
      printf("%u ", hash1);
    }
  }

  printf("\n");

  u32 sorting_error_count = 0;
  for (u32 i = 0; i < args.particle_count - 1; ++i) {
    u32 hash1 = point_hash(cpu_points[i], args.particle_count, args.particle_diameter);
    u32 hash2 = point_hash(cpu_points[i + 1], args.particle_count, args.particle_diameter);
    if (hash1 > hash2) {
      printf("sorting error: %u %u, index: %u %u\n", hash1, hash2, i, i + 1);
      ++sorting_error_count;
    }
  }
  printf("sorting errors: %u\n", sorting_error_count);

  printf("\n");

  printf("build hashtable\n");
  HashTable *table = new_hashtable(&info, gpu_points, args.particle_count, args.particle_count, args.particle_diameter,
                                   cudaStreamPerThread);
  u32 *cpu_table = (u32 *)malloc(sizeof(u32) * args.particle_count + 1 + sizeof(u32));
  cudaMemcpy(cpu_table, table, sizeof(float) * args.particle_count + 1 + sizeof(u32), cudaMemcpyDeviceToHost);
  if (args.particle_count <= 4096) {
    for (u32 i = 0; i < args.particle_count - 1; ++i) {
      u32 b = cpu_table[1 + i];
      u32 e = cpu_table[1 + i + 1];
      if (b != 0 && e != 0 && b != e) {
        printf("cell %u: %u, %u\n", i, b - 1, e - 1);
      }
    }
  }
  printf("hashtable done\n");
  launch_verify_hashtable(&info, gpu_points, args.particle_count, table, args.particle_diameter, cudaStreamPerThread);
  handle_cuda_error("collect `verify_hashtable` errors");
  printf("hashtable verified, errors have been printed\n");

  printf("\n");

  printf("prefix sum\n");
  u32 const arr_len = args.particle_count + 2;
  u32 *const gpu_array = (u32 *)table;
  u32 *const cpu_array = malloc(sizeof(u32) * arr_len);
  cudaMemcpy(cpu_array, gpu_array, sizeof(u32) * arr_len, cudaMemcpyDeviceToHost);
  launch_exclusive_scan(gpu_array, gpu_array, arr_len, cudaStreamPerThread);
  printf("prefix sum done\n");
  u32 *const cpu_scanned = malloc(sizeof(u32) * arr_len);
  cudaMemcpy(cpu_scanned, gpu_array, sizeof(u32) * arr_len, cudaMemcpyDeviceToHost);
  handle_cuda_result(cudaFree(table), "Main: Free table");
  u32 scan_error_count = cpu_scanned[0] != 0 ? 1 : 0;
  for (u32 i = 1; i < arr_len; ++i) {
    if (cpu_scanned[i] != cpu_scanned[i - 1] + cpu_array[i - 1]) {
      ++scan_error_count;
    }
  }
  free(cpu_array);
  free(cpu_scanned);
  printf("prefix sum errors: %u\n", scan_error_count);

  printf("\n");

  Geometry geometry = triangulate(&info, gpu_points, args.particle_count, args.room, args.particle_diameter,
                                  args.isolevel, cudaStreamPerThread);
  if (geometry.count <= 1024) {
    printf("\n printint triangles\n");
    Geometry const cpu_geometry = download_geometry(geometry, cudaStreamPerThread);
    print_triangles(cpu_geometry);
    free_geometry(cpu_geometry);
  } else {
    handle_cuda_result(cudaFree(geometry.triangles), "Main: Free triangles");
  }

  handle_cuda_result(cudaFree(gpu_points), "Main: Free points");
  printf("triangulation done\n");

  free(cpu_points);
  return 0;
}
