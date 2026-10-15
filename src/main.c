#include <raylib.h>
#include <stdio.h>
#include <stdlib.h>
#include "args/args.h"
#include "common.h"
#include "streamer/streamer.h"
#include "utils/utils.h"
#include "viewer/viewer.h"

static int player(PlayerArguments args, GPUInfo info);
static void random_points(RandomPointsArguments args, GPUInfo info);
static int test(GeneratorArguments args, GPUInfo info);
extern char const player_controls[]; // these must be extern due to MSVC
extern char const random_points_controls[];

int main(int const argc, char **const argv) {
  CommonArguments const default_common = (CommonArguments){.particle_diameter = 0.06f, .isolevel = 0.1f};
  GeneratorArguments const default_generator = (GeneratorArguments){
      // .common will be initialised by the parser
      // .common = default_common,
      .seed = 549951,
      .particle_count = 200,
      .room = {6.0f, 3.0f, 4.0f},
  };
  RendererArguments const default_renderer =
      (RendererArguments){.framerate = 60, .frameskip = 0, .fragment_shader_file = NULL, .vertex_shader_file = NULL};

  ProgramArguments args = (ProgramArguments){
      .mode = MODE_NONE,
      .test = default_generator,
      .common = default_common,
      .random_points =
          (RandomPointsArguments){
                                  .generator = default_generator,
                                  .renderer = default_renderer,
                                  },
      .player =
          (PlayerArguments){
                                  // .common will be initialised by the parser
              .file = "pysrc/res/fluid/test1.bin",
                                  .start_frame = 0,
                                  .total_frames = 0,
                                  .renderer = default_renderer,
                                  },
  };

  /* Parse all command line arguments into the `args` struct */
  argp_parse(&main_argp, argc, argv, ARGP_IN_ORDER, NULL, &args);

  /* Execution strictly occurs after parsing is complete */
  GPUInfo const info = fetch_gpu_info();
  printf("info:\n\tmultiprocessors: %u\n\tcores per multiprocessor: %u\n", info.multiprocessors,
         info.cores_per_multiprocessor);

  switch (args.mode) {
  case MODE_RANDOM_POINTS:
    printf("%s%s", viewer_controls, random_points_controls);
    random_points(args.random_points, info);
    return 0;
  case MODE_TEST:
    return test(args.test, info);
  case MODE_PLAYER:
    printf("%s%s", viewer_controls, player_controls);
    return player(args.player, info);
  case MODE_NONE:
    return 1;
  }
}

char const player_controls[] = "Space: play/pause\n"
                               "Period: frame-by-frame forward\n";

static int player(PlayerArguments args, GPUInfo const info) {

  FILE *const file = fopen(args.file, "rb");
  if (!file) {
    perror("file not found");
    return -ENOENT;
  }
  Streamer *streamer = streamer_create(file, args.start_frame, args.total_frames, args.renderer.frameskip);
  // after mmap we can close the file
  fclose(file);
  if (!streamer) {
    printf("failed to create streamer\n");
    return -EPIPE;
  }

  float4 *points;
  TRAP_CUDAPI(cudaMalloc((void **)&points, ARRAY_SIZE(points, streamer->num_particles)), "main: allocate points");
  u32 active_buf = 0;

  Viewer viewer = new_viewer(streamer->room, &args.common.particle_diameter, &args.common.isolevel,
                             args.renderer.framerate, args.renderer.vertex_shader_file,
                             args.renderer.fragment_shader_file, streamer->total_frames + args.start_frame);

  bool playing = true;
  bool fbf = false; // frame by frame
  u32 frame_idx = 0;

  while (!WindowShouldClose()) {
    if (playing) {
      float3 *float3_points = streamer_acquire_next_frame(streamer, active_buf, cudaStreamPerThread);
      launch_preprocess_float3(&info, float3_points, points, streamer->num_particles, cudaStreamPerThread);
      streamer_release_frame(streamer, active_buf, cudaStreamPerThread);
      active_buf = 1 - active_buf;
      frame_idx = (frame_idx + 1 + streamer->frameskip) % streamer->total_frames;
    }

    float const *const fields =
        launch_compute_scalar_fields(&info, points, streamer->num_particles, streamer->room,
                                     args.common.particle_diameter, args.common.isolevel, cudaStreamPerThread);

    Geometry const geometry =
        launch_marching_cubes(fields, streamer->num_particles, streamer->room, args.common.particle_diameter,
                              args.common.isolevel, cudaStreamPerThread);

    present(&viewer, geometry, args.start_frame + frame_idx, cudaStreamPerThread);

    set_window_title("CUDA Marching Cubes | Particle size %g | Isolevel %g | Room %gx%gx%g | Triangles: %u | FPS: %d",
                     (double)args.common.particle_diameter, (double)args.common.isolevel, (double)streamer->room.x,
                     (double)streamer->room.y, (double)streamer->room.z, geometry.count, GetFPS());
    if (fbf) {
      playing = false;
    }
    if (IsKeyPressed(KEY_SPACE)) {
      fbf = false;
      playing = !playing; // NOLINT(readability-implicit-bool-conversion)
    }
    if (IsKeyPressed(KEY_PERIOD)) {
      fbf = true;
      playing = true;
    }
  }

  CloseWindow();
  unload_viewer(&viewer);
  streamer_destroy(streamer);
  TRAP_CUDAPI(cudaFree(points), "main: free points");
  return 0;
}

char const random_points_controls[] = "R: randomize points\n"
                                      "Period: frame-by-frame randomize points\n";

static void random_points(RandomPointsArguments args, GPUInfo const info) {
  printf("seed %u\n", args.generator.seed);

  float4 *points;
  TRAP_CUDAPI(cudaMalloc((void **)&points, ARRAY_SIZE(points, args.generator.particle_count)), "main: allocate points");

  Viewer viewer =
      new_viewer(args.generator.room, &args.generator.common.particle_diameter, &args.generator.common.isolevel,
                 args.renderer.framerate, args.renderer.vertex_shader_file, args.renderer.fragment_shader_file, 1);
  bool playing = true;
  bool fbf = true; // frame by frame
  u32 frame_idx = 0;

  while (!WindowShouldClose()) {
    if (playing && (fbf || frame_idx % (args.renderer.frameskip + 1) == 0)) {
      args.generator.seed = (u32)rand() | ((u32)rand() << 16); // NOLINT(cert-msc50-cpp)
      launch_generate_points(&info, args.generator.room, args.generator.seed, points, args.generator.particle_count,
                             cudaStreamPerThread);
    }
    Geometry const gpu_geometry = launch_triangulate(&info, points, args.generator.particle_count, args.generator.room,
                                                     args.generator.common.particle_diameter,
                                                     args.generator.common.isolevel, cudaStreamPerThread);
    present(&viewer, gpu_geometry, 0, cudaStreamPerThread);
    set_window_title("CUDA Marching Cubes | Particle size %g | Isolevel %g | Room %gx%gx%g | Triangles: %u | FPS: %d",
                     (double)args.generator.particle_count, (double)args.generator.common.isolevel,
                     (double)args.generator.room.x, (double)args.generator.room.y, (double)args.generator.room.z,
                     gpu_geometry.count, GetFPS());

    if (fbf) {
      playing = false;
    }
    if (IsKeyPressed(KEY_SPACE)) {
      fbf = false;
      playing = !playing; // NOLINT(readability-implicit-bool-conversion)
    }
    if (IsKeyPressed(KEY_PERIOD)) {
      fbf = true;
      playing = true;
    }
    ++frame_idx;
  }

  CloseWindow();
  unload_viewer(&viewer);
  TRAP_CUDAPI(cudaFree(points), "main: free points");
}

static int test(GeneratorArguments const args, GPUInfo const info) {
  printf("seed %u\n", args.seed);

  float4 *const cpu_points = malloc(ARRAY_SIZE(cpu_points, args.particle_count));
  generate_random_points(cpu_points, args.particle_count, args.seed, args.room);

  if (args.particle_count <= 4096) {
    printf("\n printing unsorted points\n");
    for (u32 i = 0; i < args.particle_count; ++i) {
      u32 const hash1 = point_hash(cpu_points[i], args.particle_count, args.common.particle_diameter);
      printf("%u ", hash1);
    }
  }

  float4 *gpu_points;
  TRAP_CUDAPI(cudaMalloc((void **)&gpu_points, ARRAY_SIZE(cpu_points, args.particle_count)), "main: allocate points");
  TRAP_CUDAPI(cudaMemcpy(gpu_points, cpu_points, ARRAY_SIZE(cpu_points, args.particle_count), cudaMemcpyHostToDevice),
              "main: load points");
  printf("points loaded\n");

  printf("\n");

  printf("sort points\n");
  launch_sort_points(&info, gpu_points, args.particle_count, args.common.particle_diameter, cudaStreamPerThread);
  printf("done\n");

  // Copy result back and clean up
  TRAP_CUDAPI(cudaMemcpy(cpu_points, gpu_points, ARRAY_SIZE(gpu_points, args.particle_count), cudaMemcpyDeviceToHost),
              "main: unload points");

  if (args.particle_count <= 4096) {
    printf("\n printing sorted points\n");
    for (u32 i = 0; i < args.particle_count; ++i) {
      u32 const hash1 = point_hash(cpu_points[i], args.particle_count, args.common.particle_diameter);
      printf("%u ", hash1);
    }
  }

  printf("\n");

  u32 sorting_error_count = 0;
  for (u32 i = 0; i < args.particle_count - 1; ++i) {
    u32 const hash1 = point_hash(cpu_points[i], args.particle_count, args.common.particle_diameter);
    u32 const hash2 = point_hash(cpu_points[i + 1], args.particle_count, args.common.particle_diameter);
    if (hash1 > hash2) {
      printf("sorting error: %u %u, index: %u %u\n", hash1, hash2, i, i + 1);
      ++sorting_error_count;
    }
  }
  printf("sorting errors: %u\n", sorting_error_count);

  printf("\n");

  printf("build hashtable\n");
  HashTable *const table = new_hashtable(&info, gpu_points, args.particle_count, args.particle_count,
                                         args.common.particle_diameter, cudaStreamPerThread);
  u32 *const cpu_table = (u32 *)malloc(sizeof(u32) * args.particle_count + 1 + sizeof(u32));
  // we can't use ARRAY_SIZE here, because HashTable is an incomplete type
  TRAP_CUDAPI(
      cudaMemcpy(cpu_table, table, sizeof(float) * args.particle_count + 1 + sizeof(u32), cudaMemcpyDeviceToHost),
      "main: download table");
  if (args.particle_count <= 4096) {
    for (u32 i = 0; i < args.particle_count - 1; ++i) {
      u32 const b = cpu_table[1 + i];
      u32 const e = cpu_table[1 + i + 1];
      if (b != 0 && e != 0 && b != e) {
        printf("cell %u: %u, %u\n", i, b - 1, e - 1);
      }
    }
  }
  printf("hashtable done\n");
  launch_verify_hashtable(&info, gpu_points, args.particle_count, table, args.common.particle_diameter,
                          cudaStreamPerThread);
  printf("hashtable verified, errors have been printed\n");

  printf("\n");

  printf("prefix sum\n");
  u32 const arr_len = args.particle_count + 2;
  u32 *const gpu_array = (u32 *)table;
  u32 *const cpu_array = malloc(ARRAY_SIZE(gpu_array, arr_len));
  TRAP_CUDAPI(cudaMemcpy(cpu_array, gpu_array, ARRAY_SIZE(gpu_array, arr_len), cudaMemcpyDeviceToHost),
              "main: download array before prefix scan");
  launch_exclusive_scan(gpu_array, gpu_array, arr_len, cudaStreamPerThread);
  printf("prefix sum done\n");
  u32 *const cpu_scanned = malloc(ARRAY_SIZE(gpu_array, arr_len));
  TRAP_CUDAPI(cudaMemcpy(cpu_scanned, gpu_array, ARRAY_SIZE(gpu_array, arr_len), cudaMemcpyDeviceToHost),
              "main: download prefix scanned array");
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

  Geometry const geometry =
      launch_triangulate(&info, gpu_points, args.particle_count, args.room, args.common.particle_diameter,
                         args.common.isolevel, cudaStreamPerThread);
  if (geometry.count <= 1024) {
    printf("\n printint triangles\n");
    Geometry const cpu_geometry = download_geometry(geometry, cudaStreamPerThread);
    print_triangles(cpu_geometry);
    free_geometry(cpu_geometry);
  } else {
    TRAP_CUDAPI(cudaFree(geometry.triangles), "main: free triangles");
  }

  TRAP_CUDAPI(cudaFree(gpu_points), "main: free points");
  printf("triangulation done\n");

  free(cpu_points);
  return 0;
}
