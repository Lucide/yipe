#include <raylib.h>
#include "common.h"

typedef struct Viewer {
  float *const particle_diameter;
  float *const isolevel;
  Vector3 const center;
  float3 const room;
  s32 view_position_location;
  u32 mesh_capacity;
  Mesh mesh;
  Material material;
  Camera3D camera;
  u32 : 32;
} Viewer;

typedef enum Ownership {
  OWNERSHIP_OWNED,
  OWNERSHIP_BORROWED,
} Ownership;

Viewer new_viewer(float3 room,
                  float *particle_diameter,
                  float *isolevel,
                  u32 framerate,
                  char const *vertex_shader_path,
                  char const *fragment_shader_path);

void unload_viewer(Viewer *viewer);

// Ownership render(Viewer *viewer, Geometry const *geometry);

void present(Viewer *viewer, Geometry gpu_geometry, cudaStream_t stream);

#if defined(__GNUC__) || defined(__clang__)
#define PRINTF_LIKE(format_index, argument_index) __attribute__((format(printf, format_index, argument_index)))
#else
#define PRINTF_LIKE(format_index, argument_index)
#endif

void set_window_title(char const *format, ...) PRINTF_LIKE(1, 2);

void draw_popup(char const *format, ...) PRINTF_LIKE(1, 2);

extern char const viewer_controls[];
