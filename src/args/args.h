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

typedef struct RenderArguments {
  CommonArguments common;
  u32 seed;
  u32 particle_count;
  float3 room;
} RenderArguments;

typedef struct RandomPointsArguments {
  RenderArguments render;
  u32 framerate;
} RandomPointsArguments;

typedef RenderArguments TestArguments;

typedef struct PlayerArguments {
  CommonArguments common;
  char const *file;
  u32 framerate;
} PlayerArguments;

typedef struct ProgramArguments {
  Mode mode;
  CommonArguments common;
  RandomPointsArguments random_points;
  TestArguments test;
  PlayerArguments player;
} ProgramArguments;

extern struct argp const main_argp;
