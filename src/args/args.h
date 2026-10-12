#pragma once
#include <argp.h>
#include "common.h"

typedef enum {
  MODE_NONE = 0,
  MODE_RANDOM_POINTS,
  MODE_TEST,
  MODE_PLAYER,
} Mode;

typedef struct {
  float particle_diameter;
  float isolevel;
} CommonArguments;

typedef struct GeneratorArguments {
  CommonArguments common;
  u32 seed;
  u32 particle_count;
  float3 room;
} GeneratorArguments;

typedef struct RendererArguments {
  char const *vertex_shader_file, *fragment_shader_file;
  u32 framerate;
} RendererArguments;

typedef struct RandomPointsArguments {
  GeneratorArguments generator;
  RendererArguments renderer;
} RandomPointsArguments;

typedef GeneratorArguments TestArguments;

typedef struct PlayerArguments {
  CommonArguments common;
  char const *file;
  u32 start_frame;
  u32 total_frames;
  u8 frameskip;
  RendererArguments renderer;
} PlayerArguments;

typedef struct ProgramArguments {
  Mode mode;
  CommonArguments common;
  RandomPointsArguments random_points;
  TestArguments test;
  PlayerArguments player;
} ProgramArguments;

extern struct argp const main_argp;
