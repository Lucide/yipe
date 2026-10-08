#pragma once

#include <cuda_runtime.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <threads.h>
#include "common.h"
#include "mmap.h"

#pragma pack(push, 1)
typedef struct StreamHeader {
  float3 room;
  u32 total_frames;
  u32 num_particles;
} StreamHeader;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct Stream {
  StreamHeader header;
  float3 points[];
} Stream;
#pragma pack(pop)

typedef struct AsyncFluidStreamer {
  MappedFile mapped_file;
  StreamHeader header;
  u32 frame_bytes;

  // Double device buffers
  float4 *d_buffers[2];

  // CUDA Streams and Events
  cudaStream_t transfer_stream;
  cudaEvent_t frame_ready_events[2];
  cudaEvent_t frame_done_events[2];

  // C11 Threading & Synchronization
  thrd_t worker_thread;
  atomic_bool stop_requested;

  bool buffer_ready[2];
  bool frame_done_recorded[2];
  u32 frame_loaded[2];
} AsyncFluidStreamer;

// function declarations
AsyncFluidStreamer *streamer_create(FILE *file);
void streamer_destroy(AsyncFluidStreamer *streamer);
float4 *streamer_acquire_next_frame(AsyncFluidStreamer *streamer, u32 buf_idx, cudaStream_t compute_stream);
void streamer_release_frame(AsyncFluidStreamer *streamer, u32 buf_idx, cudaStream_t compute_stream);
