#pragma once

#include <cuda_runtime.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <threads.h>
#include "common.h"

#pragma pack(push, 1)
typedef struct StreamHeader {
  float3 room;
  u32 total_frames;
  u32 num_particles;
} StreamHeader;
#pragma pack(pop)

typedef struct AsyncFluidStreamer {
  FILE *file_stream;
  // float3 room;
  // u32 num_particles;
  // u32 total_frames;
  StreamHeader header;
  u32 file_frame_bytes;
  u32 frame_bytes;
  u32 data_offset;

  // Double buffers (Host Pinned & Device VRAM)
  float3 *h_file_buffers[2];
  float4 *h_pinned[2];
  float4 *d_buffers[2];

  // CUDA Streams and Events
  cudaStream_t transfer_stream;
  cudaEvent_t frame_ready_events[2];
  cudaEvent_t frame_done_events[2];

  // C11 Threading & Synchronization
  thrd_t worker_thread;
  atomic_bool stop_requested;

  mtx_t mutex;
  cnd_t cv_produce;
  cnd_t cv_consume;

  bool buffer_ready[2];
  bool frame_done_recorded[2];
  u32 frame_loaded[2];
} AsyncFluidStreamer;

// function declarations
AsyncFluidStreamer *streamer_create(char const *filepath);
void streamer_destroy(AsyncFluidStreamer *streamer);
float4 *streamer_acquire_next_frame(AsyncFluidStreamer *streamer, u32 buf_idx, cudaStream_t compute_stream);
void streamer_release_frame(AsyncFluidStreamer *streamer, u32 buf_idx, cudaStream_t compute_stream);
