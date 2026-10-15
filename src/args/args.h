#pragma once
#include <argp.h>
#include "common.h"

typedef enum {
  MODE_NONE = 0,
  MODE_RANDOM_POINTS,
  MODE_TEST,
  MODE_PLAYER,
} Mode;

typedef struct CommonArguments {
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
  u8 frameskip;
  STRUCT_PADDING(3);
} RendererArguments;

typedef struct RandomPointsArguments {
  RendererArguments renderer;
  GeneratorArguments generator;
  STRUCT_PADDING(4);
} RandomPointsArguments;

typedef GeneratorArguments TestArguments;

typedef struct PlayerArguments {
  RendererArguments renderer;
  char const *file;
  CommonArguments common;
  u32 start_frame;
  u32 total_frames;
} PlayerArguments;

typedef struct ProgramArguments {
  RandomPointsArguments random_points;
  PlayerArguments player;
  CommonArguments common;
  TestArguments test;
  Mode mode;
} ProgramArguments;

extern struct argp const main_argp;
