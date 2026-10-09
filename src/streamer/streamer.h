#pragma once

#include <stdatomic.h>
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
  float3 *buffers[2];

  /*
   * cuStreamWaitValue / cuStreamWriteValue synchronisation.
   *
   * Each buffer slot has two flags living in CUDA host-pinned memory
   * (visible to both CPU and GPU via their device-pointer):
   *
   *   upload_done[i]  – written by the CPU worker  after cudaMemcpy finishes,
   *                     waited on by the GPU compute stream before it reads
   *                     the buffer.
   *   gpu_done[i]     – written by the GPU compute stream when it is finished
   *                     with the buffer, waited on by the CPU worker before it
   *                     starts the next upload into that slot.
   *
   * Both are monotonically increasing sequence numbers so the wait/write
   * pairs never see stale values from a previous frame.
   */
  u32 volatile *upload_done[2]; // host-pinned, device-mapped
  u32 volatile *gpu_done[2];    // host-pinned, device-mapped
  u32 consumer_seq[2];          // deterministic sequence tracking for consumer

  /* Device pointers for cuStreamWaitValue32 / cuStreamWriteValue32.
   * Typed as unsigned long long to avoid pulling <cuda.h> into the header
   * (CUdeviceptr is unsigned long long on all platforms). */
  unsigned long long upload_done_dptr[2];
  unsigned long long gpu_done_dptr[2];

  // C11 Threading & Synchronization
  cudaStream_t worker_stream;
  thrd_t worker_thread;
  atomic_bool stop_requested;
} AsyncFluidStreamer;

// function declarations
AsyncFluidStreamer *streamer_create(FILE *file);
void streamer_destroy(AsyncFluidStreamer *streamer);
float3 *streamer_acquire_next_frame(AsyncFluidStreamer *streamer, u32 buf_idx, cudaStream_t compute_stream);
void streamer_release_frame(AsyncFluidStreamer *streamer, u32 buf_idx, cudaStream_t compute_stream);
