#include "viewer.h"
#include <math.h>
#include <rcamera.h>
#include <rlgl.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>
#include "utils/utils.h"

Viewer new_viewer(float3 const room, float *const particle_diameter, float *const isolevel, u32 const framerate) {
  SetConfigFlags(FLAG_VSYNC_HINT);
  InitWindow(1024, 768, "");
  SetTargetFPS((int)framerate);
  Vector3 const center = (Vector3){room.x / 2.0f, room.y / 2.0f, room.z / 2.0f};
  Viewer viewer = (Viewer){
      .particle_diameter = particle_diameter,
      .isolevel = isolevel,
      .center = center,
      .room = room,
      .camera =
          (Camera3D){
                     .target = center,
                     // Position the camera to fit the room in view
              .position = (Vector3){room.x * -1.f, room.y * -1.f, room.z * 1.5f},
                     .up = (Vector3){0.0f, 1.0f, .0f}, // Z is up
              .fovy = 45.0f,
                     .projection = CAMERA_PERSPECTIVE,
                     },
      .model = LoadModelFromMesh((Mesh){0}
          ),
      .shader = LoadShader("src/viewer/plastic.vs", "src/viewer/plastic.fs"),
  };

  viewer.view_position_location = GetShaderLocation(viewer.shader, "viewPos");
  if (viewer.shader.id > 0 && viewer.model.materialCount > 0) {
    viewer.model.materials[0].shader = viewer.shader;
  }
  return viewer;
}

void unload_viewer(Viewer *const viewer) {
  UnloadModel(viewer->model);
  UnloadShader(viewer->shader);
}

static Ownership update_model(Model const *const model, u32 *const triangle_capacity, Geometry const geometry) {
  Mesh *const mesh = model->meshes;
  if (*triangle_capacity > 0) {
    if (geometry.count <= *triangle_capacity) {
      mesh->vertexCount = (int)geometry.count * 3;
      mesh->triangleCount = (int)geometry.count;
      UpdateMeshBuffer(*mesh, 0, geometry.triangles, (int)mesh->triangleCount * (int)sizeof(Triangle), 0);
      return OWNERSHIP_BORROWED;
    }
    UnloadMesh(*mesh);
  } else if (geometry.count == 0) {
    return OWNERSHIP_OWNED;
  }
  *triangle_capacity = geometry.count;
  *mesh = (Mesh){
      .vertexCount = (int)*triangle_capacity * 3,
      .triangleCount = (int)*triangle_capacity,
      .vertices = (float *)geometry.triangles,
  };
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
  Ownership const ownership = update_model(&viewer->model, &viewer->mesh_capacity, *geometry);
  // bool const owned = false;

  update_parameters(viewer);
  update_camera(viewer);

  ClearBackground(RAYWHITE);

  BeginDrawing();

  BeginMode3D(viewer->camera);
  // rlDisableBackfaceCulling();

  DrawCubeWires(viewer->center, viewer->room.x, viewer->room.y, viewer->room.z, DARKGRAY);

  if (viewer->shader.id > 0) {
    SetShaderValue(viewer->shader, viewer->view_position_location, &viewer->camera.position, SHADER_UNIFORM_VEC3);
  }
  DrawModel(viewer->model, (Vector3){0}, 1.0f, RED);
  // DrawModelWires(viewer->model, (Vector3){0}, 1.0f, BLACK);
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

  if (difftime(now, last_update) < 1.0) {
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
