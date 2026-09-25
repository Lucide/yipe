#include <raylib.h>
#include <stdbool.h>
#include "common.h"

typedef struct Viewer {
  Vector3 const origin;
  Camera3D camera;
  Mesh mesh;
  Model model;
  u32 mesh_capacity;
  u32 : 32;
} Viewer;

typedef enum Ownership {
  OWNERSHIP_OWNED,
  OWNERSHIP_BORROWED,
} Ownership;

Viewer new_viewer(void);

Ownership render(Viewer *viewer, Geometry const *geometry);
