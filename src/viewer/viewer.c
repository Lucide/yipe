#include "viewer.h"
#include <math.h>
#include <rcamera.h>
#include <rlgl.h>

Viewer new_viewer(void) {
  SetConfigFlags(FLAG_VSYNC_HINT);
  InitWindow(1024, 768, "");
  SetTargetFPS(60);
  Vector3 const origin = (Vector3){ROOM_WIDTH / 2.0f, ROOM_DEPTH / 2.0f, ROOM_HEIGHT / 2.0f};
  return (Viewer){
      .origin = origin,
      .camera =
          (Camera3D){
                     .target = origin,
                     // Position the camera to fit the room in view
              .position = (Vector3){ROOM_WIDTH * -1.f, ROOM_DEPTH * -1.f, ROOM_HEIGHT * 1.5f},
                     .up = (Vector3){0.0f, 0.0f, 1.0f}, // Z is up
              .fovy = 45.0f,
                     .projection = CAMERA_PERSPECTIVE,
                     },
      .model = LoadModelFromMesh((Mesh){0}
          )
  };
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

static void update_camera(Camera3D *const camera) {
  Vector2 const delta = GetMouseDelta();
  if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
    CameraYaw(camera, -delta.x * 0.003f, true);
    CameraPitch(camera, -delta.y * 0.003f, true, true, false);
  }
  float const room_scale = sqrtf(ROOM_WIDTH * ROOM_WIDTH + ROOM_DEPTH * ROOM_DEPTH + ROOM_HEIGHT * ROOM_HEIGHT);

  if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
    float const pan_scale = room_scale * 0.002f;

    CameraMoveRight(camera, -delta.x * pan_scale, false);
    CameraMoveUp(camera, delta.y * pan_scale);
  }

  float const zoom_step = room_scale * 0.04f;
  CameraMoveToTarget(camera, -GetMouseWheelMove() * zoom_step);
}

static void draw_room_dimensions(Viewer const *const viewer) {
  // float const half_width = ROOM_WIDTH * 0.5f;
  // float const half_depth = ROOM_DEPTH * 0.5f;
  // float const half_height = ROOM_HEIGHT * 0.5f;

  Vector3 const width_point = {
      ROOM_WIDTH,
      0,
      0,
  };

  Vector3 const depth_point = {
      0,
      ROOM_DEPTH,
      0,
  };

  Vector3 const height_point = {
      0,
      0,
      ROOM_HEIGHT,
  };

  Vector2 const origin_screen = GetWorldToScreen((Vector3){0}, viewer->camera);
  Vector2 const width_screen = GetWorldToScreen(width_point, viewer->camera);
  Vector2 const depth_screen = GetWorldToScreen(depth_point, viewer->camera);
  Vector2 const height_screen = GetWorldToScreen(height_point, viewer->camera);

  DrawText("0", (int)origin_screen.x, (int)origin_screen.y, 15, DARKGRAY);

  DrawText(TextFormat("%d", ROOM_WIDTH), (int)width_screen.x, (int)width_screen.y, 15, DARKGRAY);
  DrawText(TextFormat("%d", ROOM_HEIGHT), (int)height_screen.x, (int)height_screen.y, 15, DARKGRAY);
  DrawText(TextFormat("%d", ROOM_DEPTH), (int)depth_screen.x, (int)depth_screen.y, 15, DARKGRAY);
}

Ownership render(Viewer *const viewer, Geometry const *const geometry) {
  Ownership const ownership = update_model(&viewer->model, &viewer->mesh_capacity, *geometry);
  // bool const owned = false;

  update_camera(&viewer->camera);

  ClearBackground(RAYWHITE);

  BeginDrawing();

  BeginMode3D(viewer->camera);
  // rlDisableBackfaceCulling();

  DrawCubeWires(viewer->origin, ROOM_WIDTH, ROOM_DEPTH, ROOM_HEIGHT, DARKGRAY);

  DrawModel(viewer->model, (Vector3){0}, 1.0f, RED);
  DrawModelWires(viewer->model, (Vector3){0}, 1.0f, BLACK);
  EndMode3D();

  draw_room_dimensions(viewer);

  EndDrawing();

  return ownership;
}
