#include "args.h"
#include <stdlib.h>
#include <string.h>

static char const main_doc[] = "yipe -- A onesweep/spatial-hashing/marching-cubes demo.\n\n"
                               "Subcommands:\n"
                               "  random          Render random points demo\n"
                               "  test            Run test suite\n"
                               "  player          Run player mode";

/* -------------------------------------------------------------------------- */
/* Common / Shared Options                                                     */
/* -------------------------------------------------------------------------- */

static struct argp_option const common_options[] = {
    {"particle-diameter", 'd', "NUM", 0,          "Diameter of the particles", 0},
    {         "isolevel", 'l', "NUM", 0, "Isolevel for generating isosurface", 0},
    {               NULL,   0,  NULL, 0,                                 NULL, 0}
};

static error_t parse_common_opt(int key, char *arg, struct argp_state *state) {
  CommonArguments *common = state->input;
  char *endptr;

  switch (key) {
  case 'd': {
    float const diameter = strtof(arg, &endptr);
    if (arg == endptr || *endptr != '\0') {
      argp_failure(state, 1, 0, "Invalid argument: \"%s\"; particle diameter must be a number", arg);
    }
    common->particle_diameter = diameter;
    break;
  }
  case 'i': {
    float const isolevel = strtof(arg, &endptr);
    if (arg == endptr || *endptr != '\0') {
      argp_failure(state, 1, 0, "Invalid argument: \"%s\"; isolevel must be a number", arg);
    }
    common->isolevel = isolevel;
    break;
  }
  case ARGP_KEY_END:
    if (common->isolevel <= 0) {
      argp_failure(state, 1, 0, "Isolevel must be greater than 0");
    }
    if (common->particle_diameter <= 0) {
      argp_failure(state, 1, 0, "Particle diameter must be greater than 0");
    }
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static error_t parse_main_common_opt(int key, char *arg, struct argp_state *state) {
  if (key == ARGP_KEY_END) {
    return 0; /* Defer required option validation to the subcommand parser */
  }
  return parse_common_opt(key, arg, state);
}

static struct argp const common_argp = {common_options, parse_common_opt, NULL, NULL, NULL, NULL, NULL};
static struct argp const main_common_argp = {common_options, parse_main_common_opt, NULL, NULL, NULL, NULL, NULL};

static struct argp_child const main_children[] = {
    {&main_common_argp, 0, "Common Options:", 0},
    {0}
};

/* -------------------------------------------------------------------------- */
/* Generator Options (Shared base for test / random)                      */
/* -------------------------------------------------------------------------- */

static struct argp_option const generator_options[] = {
    {          "seed", 's', "NUM", 0, "Seed for random points", 0},
    {"particle-count", 'c', "NUM", 0,    "Number of particles", 0},
    {    "room-width", 'x', "NUM", 0,   "Room width in metres", 0},
    {   "room-height", 'y', "NUM", 0,  "Room height in metres", 0},
    {    "room-depth", 'z', "NUM", 0,   "Room depth in metres", 0},
    {            NULL,   0,  NULL, 0,                     NULL, 0}
};

static struct argp_child const generator_children[] = {
    {&common_argp, 0, "Shared Options:", 0},
    {0}
};

static error_t parse_generator_opt(int key, char *arg, struct argp_state *state) {
  GeneratorArguments *arguments = state->input;
  char *endptr;

  switch (key) {
  case ARGP_KEY_INIT:
    /* Map child_inputs[0] so common_argp writes directly into arguments->common */
    state->child_inputs[0] = &arguments->common;
    break;
  case 's': {
    u32 const seed = strtoul(arg, &endptr, 0);
    if (arg == endptr || *endptr != '\0') {
      argp_failure(state, 1, 0, "Invalid argument: \"%s\"; seed must be an integer", arg);
    }
    arguments->seed = seed;
    break;
  }
  case 'c': {
    u32 const particle_count = strtoul(arg, &endptr, 0);
    if (arg == endptr || *endptr != '\0') {
      argp_failure(state, 1, 0, "Invalid argument: \"%s\"; particle count must be an integer", arg);
    }
    arguments->particle_count = particle_count;
    break;
  }
  case 'x': {
    float const x = strtof(arg, &endptr);
    if (arg == endptr || *endptr != '\0') {
      argp_failure(state, 1, 0, "Invalid argument: \"%s\"; room width must be a number", arg);
    }
    arguments->room.x = x;
    break;
  }
  case 'y': {
    float const y = strtof(arg, &endptr);
    if (arg == endptr || *endptr != '\0') {
      argp_failure(state, 1, 0, "Invalid argument: \"%s\"; room height must be a number", arg);
    }
    arguments->room.y = y;
    break;
  }
  case 'z': {
    float const z = strtof(arg, &endptr);
    if (arg == endptr || *endptr != '\0') {
      argp_failure(state, 1, 0, "Invalid argument: \"%s\"; room depth must be a number", arg);
    }
    arguments->room.z = z;
    break;
  }
  case ARGP_KEY_ARG:
    argp_error(state, "Unexpected positional argument: %s", arg);
    break;
  case ARGP_KEY_END:
    if (arguments->room.x <= 0 || arguments->room.y <= 0 || arguments->room.z <= 0) {
      argp_failure(state, 1, 0, "All room dimensions must be greater than 0");
    }
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp const generator_argp = {
    generator_options, parse_generator_opt, NULL, NULL, generator_children, NULL, NULL};

/* -------------------------------------------------------------------------- */
/* Random Points Subcommand                                                   */
/* -------------------------------------------------------------------------- */

static struct argp_option const random_points_options[] = {
    {      "framerate", 'f',  "NUM", 0,     "Framerate in frames per second", 0},
    {      "frameskip", 'k',  "NUM", 0, "Skipped frames (0 for no skipping)", 0},
    {  "vertex-shader", 'V', "FILE", 0,         "Path to vertex shader file", 0},
    {"fragment-shader", 'F', "FILE", 0,       "Path to fragment shader file", 0},
    {             NULL,   0,   NULL, 0,                                 NULL, 0}
};

static struct argp_child const random_points_children[] = {
    {&generator_argp, 0, NULL, 0},
    {0}
};

static error_t parse_random_points_opt(int key, char *arg, struct argp_state *state) {
  RandomPointsArguments *arguments = state->input;
  char *endptr;

  switch (key) {
  case ARGP_KEY_INIT:
    /* Direct render options to arguments->render */
    state->child_inputs[0] = &arguments->generator;
    break;
  case 'f': {
    u32 const framerate = strtoul(arg, &endptr, 0);
    if (arg == endptr || *endptr != '\0') {
      argp_failure(state, 1, 0, "Invalid argument: \"%s\"; framerate must be an integer", arg);
    }
    arguments->renderer.framerate = framerate;
    break;
  }
  case 'k': {
    u32 const frameskip = strtoul(arg, &endptr, 0);
    if (arg == endptr || *endptr != '\0') {
      argp_failure(state, 1, 0, "Invalid argument: \"%s\"; frameskip must be a number", arg);
    }
    if (frameskip >= 256) {
      argp_failure(state, 1, 0, "Invalid argument: frameskip must be less than 256");
    }
    arguments->renderer.frameskip = (u8)frameskip;
    break;
  }
  case 'V': {
    arguments->renderer.vertex_shader_file = arg;
    break;
  }
  case 'F': {
    arguments->renderer.fragment_shader_file = arg;
    break;
  }
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp const random_points_argp = {
    random_points_options, parse_random_points_opt, NULL, NULL, random_points_children, NULL, NULL};

/* -------------------------------------------------------------------------- */
/* Player Subcommand                                                          */
/* -------------------------------------------------------------------------- */

static struct argp_option const player_options[] = {
    {          "input", 'i', "FILE", 0,              "Path to the recording file", 0},
    {      "framerate", 'f',  "NUM", 0,                               "framerate", 0},
    {  "vertex-shader", 'V', "FILE", 0,              "Path to vertex shader file", 0},
    {"fragment-shader", 'F', "FILE", 0,            "Path to fragment shader file", 0},
    {      "frameskip", 'k',  "NUM", 0,      "Skipped frames (0 for no skipping)", 0},
    {    "start-frame", 'b',  "NUM", 0,                 "Index of starting frame", 0},
    {   "total-frames", 't',  "NUM", 0, "Max number of frames (0 for all frames)", 0},
    {             NULL,   0,   NULL, 0,                                      NULL, 0}
};

static struct argp_child const player_children[] = {
    {&common_argp, 0, "Shared Options:", 0},
    {0}
};

static error_t parse_player_opt(int key, char *arg, struct argp_state *state) {
  PlayerArguments *arguments = state->input;

  char *endptr;
  switch (key) {
  case 'f': {
    u32 const framerate = strtoul(arg, &endptr, 0);
    if (arg == endptr || *endptr != '\0') {
      argp_failure(state, 1, 0, "Invalid argument: \"%s\"; framerate must be a number", arg);
    }
    arguments->renderer.framerate = framerate;
    break;
  }
  case 'i': {
    arguments->file = arg;
    break;
  }
  case 'V': {
    arguments->renderer.vertex_shader_file = arg;
    break;
  }
  case 'F': {
    arguments->renderer.fragment_shader_file = arg;
    break;
  }
  case 'k': {
    u32 const frameskip = strtoul(arg, &endptr, 0);
    if (arg == endptr || *endptr != '\0') {
      argp_failure(state, 1, 0, "Invalid argument: \"%s\"; frameskip must be a number", arg);
    }
    if (frameskip >= 256) {
      argp_failure(state, 1, 0, "Invalid argument: frameskip must be less than 256");
    }
    arguments->renderer.frameskip = (u8)frameskip;
    break;
  }
  case 'b': {
    u32 const start_frame = strtoul(arg, &endptr, 0);
    if (arg == endptr || *endptr != '\0') {
      argp_failure(state, 1, 0, "Invalid argument: \"%s\"; start-frame must be a number", arg);
    }
    arguments->start_frame = start_frame;
    break;
  }
  case 't': {
    u32 const total_frames = strtoul(arg, &endptr, 0);
    if (arg == endptr || *endptr != '\0') {
      argp_failure(state, 1, 0, "Invalid argument: \"%s\"; total-frames must be a number", arg);
    }
    arguments->total_frames = total_frames;
    break;
  }
  case ARGP_KEY_INIT:
    state->child_inputs[0] = &arguments->common;
    break;
  case ARGP_KEY_END:
    if (arguments->file == NULL) {
      argp_failure(state, 1, 0, "Missing required argument: --file");
    }
    break;
  case ARGP_KEY_ARG:
    argp_error(state, "Unexpected positional argument: %s", arg);
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp const player_argp = {player_options, parse_player_opt, NULL, NULL, player_children, NULL, NULL};

/* -------------------------------------------------------------------------- */
/* Main Parser                                                                */
/* -------------------------------------------------------------------------- */

static error_t parse_main_opt(int key, char *arg, struct argp_state *state) {
  ProgramArguments *prog_args = state->input;

  switch (key) {
  case ARGP_KEY_INIT:
    state->child_inputs[0] = &prog_args->common;
    break;
  case ARGP_KEY_ARG:
    if (state->arg_num == 0) {
      struct argp const *sub_argp = NULL;
      void *sub_args = NULL;

      if (strcmp(arg, "random") == 0) {
        prog_args->mode = MODE_RANDOM_POINTS;
        prog_args->random_points.generator.common = prog_args->common;
        sub_argp = &random_points_argp;
        sub_args = &prog_args->random_points;
      } else if (strcmp(arg, "test") == 0) {
        prog_args->mode = MODE_TEST;
        prog_args->test.common = prog_args->common;
        sub_argp = &generator_argp;
        sub_args = &prog_args->test;
      } else if (strcmp(arg, "player") == 0) {
        prog_args->mode = MODE_PLAYER;
        prog_args->player.common = prog_args->common;
        sub_argp = &player_argp;
        sub_args = &prog_args->player;
      } else {
        argp_error(state, "Unknown subcommand: \"%s\"", arg);
      }

      /* Forward remaining arguments to subcommand parser */
      int const sub_argc = state->argc - state->next + 1;
      char **const sub_argv = &state->argv[state->next - 1];
      sub_argv[0] = arg;

      argp_parse(sub_argp, sub_argc, sub_argv, 0, NULL, sub_args);
      state->next = state->argc; /* Mark all arguments consumed */
    }
    break;
  case ARGP_KEY_END:
    if (prog_args->mode == MODE_NONE) {
      argp_error(state, "A subcommand {random|test|player} is required");
    }
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

struct argp const main_argp = {NULL, parse_main_opt, "{random|test|player} [options]", main_doc, main_children, NULL,
                               NULL};
