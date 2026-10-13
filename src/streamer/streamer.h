#pragma once

#include <cuda.h>
#include <stdatomic.h>
#include <threads.h>
#include "common.h"
#include "mmap.h"

typedef struct Streamer {
  MappedFile mapped_file;

  thrd_t worker_thread;

  u32 volatile *upload_done[2];
  u32 volatile *gpu_done[2];

  // device pointers for cuStreamWaitValue32 / cuStreamWriteValue32
  CUdeviceptr upload_done_dptr[2];
  CUdeviceptr gpu_done_dptr[2];

  u32 consumer_seq[2];

  float3 *buffers[2];

  size_t frame_size;
  float3 room;
  u32 total_frames;
  u32 num_particles;
  u32 start_frame;
  u32 end_frame;
  u8 frameskip;

  atomic_bool stop_requested;

  struct {
    u8 _pad[2];
  };
} Streamer;

#pragma pack(push, 1)
typedef struct StreamHeader {
  float3 room;
  u32 total_frames;
  u32 num_particles;
} StreamHeader;
#pragma pack(pop)

// function declarations
Streamer *streamer_create(FILE *file, u32 start_frame, u32 total_frames, u8 frameskip);
void streamer_destroy(Streamer *streamer);
float3 *streamer_acquire_next_frame(Streamer const *streamer, u32 buf_idx, cudaStream_t compute_stream);
void streamer_release_frame(Streamer *streamer, u32 buf_idx, cudaStream_t compute_stream);
