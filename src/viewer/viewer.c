#include "viewer.h"
#include <math.h>
#include <rcamera.h>
#include <rlgl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils/utils.h"

extern char const default_vertex_shader[];
extern char const default_fragment_shader[];

static Matrix const identity = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

Viewer new_viewer(float3 const room,
                  float *const particle_diameter,
                  float *const isolevel,
                  u32 const framerate,
                  char const *const vertex_shader_path,
                  char const *const fragment_shader_path) {
  SetConfigFlags(FLAG_VSYNC_HINT);
  InitWindow(1024, 768, "");
  SetTargetFPS((int)framerate);
  Vector3 const center = (Vector3){room.x / 2.0f, room.y / 2.0f, room.z / 2.0f};

  char const *vertex_shader = default_vertex_shader;
  char const *fragment_shader = default_fragment_shader;
  if (vertex_shader_path) {
    read32_file(vertex_shader_path, &vertex_shader, NULL);
  }
  if (fragment_shader_path) {
    read32_file(fragment_shader_path, &fragment_shader, NULL);
  }
  Shader const shader = LoadShaderFromMemory(vertex_shader, fragment_shader);
  if (vertex_shader != default_vertex_shader) {
    free((void *)(uintptr_t)vertex_shader);
  }
  if (fragment_shader != default_fragment_shader) {
    free((void *)(uintptr_t)fragment_shader);
  }
  Material material = LoadMaterialDefault();
  material.maps[MATERIAL_MAP_DIFFUSE].color = BLUE;
  if (IsShaderValid(shader)) {
    material.shader = shader;
  }

  Viewer viewer = (Viewer){
      .particle_diameter = particle_diameter,
      .isolevel = isolevel,
      .center = center,
      .room = room,
      .camera =
          (Camera3D){
                     .target = center,
                     // position the camera to fit the room in view
              .position = (Vector3){room.x * -1.f, room.y * -1.f, room.z * 1.5f},
                     .up = (Vector3){0.0f, 1.0f, .0f}, // Z is up
              .fovy = 45.0f,
                     .projection = CAMERA_PERSPECTIVE,
                     },
      .mesh = (Mesh){0},
      .material = material
  };

  viewer.view_position_location = GetShaderLocation(viewer.material.shader, "viewPos");
  return viewer;
}

void unload_viewer(Viewer *const viewer) {
  viewer->mesh.normals = NULL; // prevent double free
  UnloadMesh(viewer->mesh);
  UnloadMaterial(viewer->material);
}

static Ownership update_mesh(Mesh *const mesh, u32 *const triangle_capacity, Geometry const geometry) {
  if (*triangle_capacity > 0) {
    if (geometry.count <= *triangle_capacity) {
      mesh->vertexCount = (int)geometry.count * 3;
      mesh->triangleCount = (int)geometry.count;
      UpdateMeshBuffer(*mesh, 0, geometry.triangles, (int)ARRAY_SIZE(geometry.triangles, (u32)mesh->triangleCount), 0);
      UpdateMeshBuffer(*mesh, 2, geometry.normals, (int)ARRAY_SIZE(geometry.normals, (u32)mesh->triangleCount), 0);
      return OWNERSHIP_BORROWED;
    }
    mesh->normals = NULL; // prevent double free
    UnloadMesh(*mesh);
  } else if (geometry.count == 0) {
    return OWNERSHIP_BORROWED;
  }
  *triangle_capacity = geometry.count;
  *mesh = (Mesh){.vertexCount = (int)*triangle_capacity * 3,
                 .triangleCount = (int)*triangle_capacity,
                 .vertices = (float *)geometry.triangles,
                 .normals = (float *)geometry.normals};
  UploadMesh(mesh, true);
  return OWNERSHIP_OWNED;
}

static void update_parameters(Viewer *const viewer) {
  // make particles bigger (B)
  if (IsKeyPressed(KEY_LEFT)) {
    *viewer->particle_diameter += 0.01f;
  } else if (IsKeyDown(KEY_LEFT)) {
    *viewer->particle_diameter += 0.001f;
  }
  // make particles smaller (S)
  if (IsKeyPressed(KEY_RIGHT)) {
    *viewer->particle_diameter -= 0.01f;
  } else if (IsKeyDown(KEY_RIGHT)) {
    *viewer->particle_diameter -= 0.001f;
  }

  // make isolevel higher (H)
  if (IsKeyPressed(KEY_UP)) {
    *viewer->isolevel *= 1.1f;
  } else if (IsKeyDown(KEY_UP)) {
    *viewer->isolevel *= 1.01f;
  }
  // make isolevel lower (L)
  if (IsKeyPressed(KEY_DOWN)) {
    *viewer->isolevel /= 1.1f;
  } else if (IsKeyDown(KEY_DOWN)) {
    *viewer->isolevel /= 1.01f;
  }
}

char const viewer_controls[] = "Mouse Middle: pan\n"
                               "Mouse Left: rotate\n"
                               "←: increase particle size\n"
                               "→: decrease particle size\n"
                               "↑: increase isolevel\n"
                               "↓: decrease isolevel\n"
                               "W: move camera forward"
                               "A: move camera to the left\n"
                               "S: move camera backward\n"
                               "D: move camera to the right\n";

static void update_camera(Viewer *const viewer) {
  Vector2 const delta = GetMouseDelta();
  float const room_scale =
      sqrtf(viewer->room.x * viewer->room.x + viewer->room.y * viewer->room.y + viewer->room.z * viewer->room.z);

  if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
    float const pan_scale = room_scale * 0.002f;

    CameraMoveRight(&viewer->camera, -delta.x * pan_scale, false);
    CameraMoveUp(&viewer->camera, delta.y * pan_scale);
  }

  if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
    CameraYaw(&viewer->camera, -delta.x * 0.003f, true);
    CameraPitch(&viewer->camera, -delta.y * 0.003f, true, true, false);
  }

  float const zoom_step = room_scale * 0.04f;
  CameraMoveToTarget(&viewer->camera, -GetMouseWheelMove() * zoom_step);

  if (IsKeyDown(KEY_W)) {
    CameraMoveForward(&viewer->camera, room_scale * 0.002f, true);
  }
  if (IsKeyDown(KEY_A)) {
    CameraMoveRight(&viewer->camera, -room_scale * 0.002f, true);
  }
  if (IsKeyDown(KEY_S)) {
    CameraMoveForward(&viewer->camera, -room_scale * 0.002f, true);
  }
  if (IsKeyDown(KEY_D)) {
    CameraMoveRight(&viewer->camera, room_scale * 0.002f, true);
  }
}

static void draw_room_dimensions(Viewer const *const viewer) {
  Vector3 const width_point = {
      viewer->room.x,
      0,
      0,
  };

  Vector3 const height_point = {
      0,
      viewer->room.y,
      0,
  };

  Vector3 const depth_point = {
      0,
      0,
      viewer->room.z,
  };

  Vector2 const origin_screen = GetWorldToScreen((Vector3){0}, viewer->camera);
  Vector2 const width_screen = GetWorldToScreen(width_point, viewer->camera);
  Vector2 const height_screen = GetWorldToScreen(height_point, viewer->camera);
  Vector2 const depth_screen = GetWorldToScreen(depth_point, viewer->camera);

  DrawText("0", (int)origin_screen.x, (int)origin_screen.y, 15, DARKGRAY);

  DrawText(TextFormat("%g", (double)viewer->room.x), (int)width_screen.x, (int)width_screen.y, 15, DARKGRAY);
  DrawText(TextFormat("%g", (double)viewer->room.y), (int)height_screen.x, (int)height_screen.y, 15, DARKGRAY);
  DrawText(TextFormat("%g", (double)viewer->room.z), (int)depth_screen.x, (int)depth_screen.y, 15, DARKGRAY);
}

static Ownership render(Viewer *const viewer, Geometry const *const geometry) {
  Ownership const ownership = update_mesh(&viewer->mesh, &viewer->mesh_capacity, *geometry);

  update_parameters(viewer);
  update_camera(viewer);

  ClearBackground(RAYWHITE);
  BeginDrawing();
  BeginMode3D(viewer->camera);

  // rlDisableBackfaceCulling();
  DrawCubeWires(viewer->center, viewer->room.x, viewer->room.y, viewer->room.z, DARKGRAY);
  SetShaderValue(viewer->material.shader, viewer->view_position_location, &viewer->camera.position,
                 SHADER_UNIFORM_VEC3);
  DrawMesh(viewer->mesh, viewer->material, identity);

  EndMode3D();
  draw_room_dimensions(viewer);
  EndDrawing();

  return ownership;
}

void present(Viewer *const viewer, Geometry const gpu_geometry, cudaStream_t const stream) {
  Geometry geometry = download_geometry(gpu_geometry, stream);

  if (OWNERSHIP_BORROWED == render(viewer, &geometry)) {
    free_geometry(geometry);
  }
}

void set_window_title(char const *format, ...) {
  static time_t last_update;
  time_t const now = time(NULL);

  if (difftime(now, last_update) < 0.5) {
    return;
  }

  char title[256];
  va_list args;
  va_start(args, format);
  vsnprintf(title, sizeof(title), format, args);
  va_end(args);

  SetWindowTitle(title);
  last_update = now;
}

void draw_popup(char const *format, ...) {
  char title[256];
  va_list args;
  va_start(args, format);
  vsnprintf(title, sizeof(title), format, args);
  va_end(args);

  float const padding = 10.f;
  int const font_size = 20;
  Font font = GetFontDefault();
  float spacing = (float)font_size / 10.0f;
  Vector2 text_size = MeasureTextEx(font, title, (float)font_size, spacing);

  float box_width = text_size.x + (padding * 2);
  float box_height = text_size.y + (padding * 2);
  float box_x = ((float)-GetScreenWidth() - box_width) / 2.0f;
  float box_y = ((float)-GetScreenHeight() - box_height) / 2.0f;

  BeginDrawing();
  DrawRectangle((int)box_x, (int)box_y, (int)box_width, (int)box_height, (Color){80, 80, 80, 200});
  DrawText(title, (int)(box_x + padding), (int)(box_y + padding), font_size, LIGHTGRAY);
  EndDrawing();
}
