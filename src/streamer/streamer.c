#include "streamer.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector_types.h>
#ifdef _MSC_VER
#define fseek64 _fseeki64
#define ftell64 _ftelli64
#else
#define _FILE_OFFSET_BITS 64
#define fseek64 fseeko
#define ftell64 ftello
#endif
#include <stdio.h>
#include "mmap.h"

/* ------------------------------------------------------------------ */
/*  Worker thread                                                     */
/* ------------------------------------------------------------------ */

static int worker_loop(void *const arg) {
  Streamer const *const streamer = (Streamer const *)arg;
  u8 slot = 0;
  u32 seq = 0;

  // frames are increased modulo streamer->total_frames, so the animation loops
  //
  // Each slot has it's own increasing seq number
  //  slot 0 is used by frames 0, 2, 4, … → seq 1, 2, 3, ...
  //  slot 1 is used by frames 1, 3, 5, … → seq 1, 2, 3, ...
  //
  // Slot flips at every iteration, so seq increases every two iteration
  // If we start from slot 0, it's the same as summing slot
  for (u32 frame = 0; !atomic_load(&streamer->stop_requested);
       frame = (frame + streamer->frameskip + 1) % streamer->total_frames, // choose next frame
       seq += slot,                                                        // increment every two iterations
       slot = 1 - slot                                                     // flip slot
  ) {

    // waiting must happen in the cpu because the gpu can deadlock
    // spin wait for gpu_done
    while (*streamer->gpu_done[slot] < seq && !atomic_load(&streamer->stop_requested)) {
      thrd_yield();
    }

    if (atomic_load(&streamer->stop_requested))
      return 0;

    // synchronoud copy
    cudaMemcpy(streamer->buffers[slot], &((float3 const *)streamer->mapped_file.data)[streamer->num_particles * frame],
               streamer->frame_size, cudaMemcpyHostToDevice);

    // publishing must also happen in the cpu with mapped memory because the gpu can deadlock
    // publish upload_done
    *streamer->upload_done[slot] = seq + 1;
  }
  return 0;
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

// allocate one pinned+mapped uint32 flag
static int alloc_flag(u32 volatile **const host_ptr, CUdeviceptr *const dev_ptr, u32 const initial_value) {
  // allocate memory mapped flag (the cpu must be able to read and update it)
  u32 *hostptr;
  if (cudaHostAlloc((void **)&hostptr, sizeof(u32), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess)
    return -1;

  *hostptr = initial_value;

  // get device pointers for the mapped memory
  // the gpu will use this to wait/write
  CUdeviceptr deviceptr;
  if (cudaHostGetDevicePointer((void **)&deviceptr, hostptr, 0) != cudaSuccess) {
    // free the mapped memory if it failed
    cudaFreeHost(hostptr);
    return -1;
  }

  // pass back the pointers
  *host_ptr = hostptr;
  *dev_ptr = deviceptr;
  return 0;
}

static bool parse_header(
    StreamHeader *const header, FILE *const file, u32 const start_frame, u32 total_frames, u8 const frameskip) {
  size_t const bytes_read = fread(header, 1, sizeof(StreamHeader), file);
  if (bytes_read < sizeof(StreamHeader)) {
    if (ferror(file)) {
      perror("error reading recording header");
      return false;
    }
  }

  if (header->room.x <= 0 && header->room.y <= 0 && header->room.z <= 0) {
    printf("all room dimensions must be greater than 0 (found %gx%gx%g)\n", (double)header->room.x,
           (double)header->room.y, (double)header->room.z);
    return false;
  }

  if (header->total_frames == 0) {
    printf("file has no frames\n");
    return false;
  }
  if (header->num_particles == 0) {
    printf("frames are empty\n");
    return false;
  }

  // determine file size
  if (fseek64(file, 0, SEEK_END) != 0) {
    perror("error seeking file end");
    return false;
  }

  s64 const size = ftell64(file);
  if (size < 0) {
    perror("error telling file size");
    return false;
  }

  u64 const frame_size = header->num_particles * sizeof(float3);
  u64 const expected_size = sizeof(StreamHeader) + header->total_frames * frame_size;
  if ((u64)size < expected_size) {
    printf("file ends too soon, %lld bytes (expected %llu)\n", size, header->total_frames * frame_size);
    return false;
  }
  if ((u64)size > expected_size) {
    printf("file too long, %lld bytes (expected %llu)\n", size, header->total_frames * frame_size);
    return false;
  }

  if (start_frame >= header->total_frames) {
    printf("start_frame \"%u\" is outside the recording file (%u frames)\n", start_frame, header->total_frames);
    return false;
  }

  total_frames = total_frames ? min(total_frames, header->total_frames - start_frame) : header->total_frames;

  if (frameskip >= total_frames) {
    printf("frameskip (%u) skips every frame (%u)\n", frameskip, total_frames);
    return false;
  }

  header->total_frames = total_frames;
  return true;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                        */
/* ------------------------------------------------------------------ */

Streamer *streamer_create(FILE *const file, u32 const start_frame, u32 const total_frames, u8 const frameskip) {
  Streamer *streamer = malloc(sizeof(Streamer));

  StreamHeader header;
  if (!parse_header(&header, file, start_frame, total_frames, frameskip))
    return NULL;
  u32 const frame_size = header.num_particles * sizeof(float3);
  size_t const map_start = sizeof(StreamHeader) + start_frame * frame_size;
  size_t const map_size = header.total_frames * frame_size;

  *streamer = (Streamer){
      .room = header.room,
      .total_frames = header.total_frames,
      .num_particles = header.num_particles,
      .frame_size = frame_size,
      .start_frame = start_frame,
      .frameskip = frameskip,
  };

  mmap_file(file, map_start, map_size, &streamer->mapped_file);

  // allocate two device buffers (one contiguous allocation)
  float3 *buffers;
  cudaMalloc((void **)&buffers, 2 * frame_size);
  streamer->buffers[0] = buffers;
  streamer->buffers[1] = &buffers[header.num_particles];

  // allocate pinned+mapped synchronisation flags
  for (u32 i = 0; i < 2; ++i) {
    streamer->consumer_seq[i] = 1;
    if (alloc_flag(&streamer->upload_done[i], &streamer->upload_done_dptr[i], 0) != 0 ||
        alloc_flag(&streamer->gpu_done[i], &streamer->gpu_done_dptr[i], 0) != 0) {
      streamer_destroy(streamer);
      return NULL;
    }
  }

  atomic_init(&streamer->stop_requested, false);

  // launch background worker thread
  if (thrd_create(&streamer->worker_thread, worker_loop, streamer) != thrd_success) {
    atomic_store(&streamer->stop_requested, true);
    streamer_destroy(streamer);
    return NULL;
  }

  return streamer;
}

void streamer_destroy(Streamer *const streamer) {
  if (!streamer)
    return;

  // signal the worker to stop and join it
  if (!atomic_load(&streamer->stop_requested)) {
    atomic_store(&streamer->stop_requested, true);
    // unblock the worker thread if it's waiting on cudaStreamSynchronize
    for (u32 i = 0; i < 2; ++i) {
      if (streamer->gpu_done[i]) {
        *streamer->gpu_done[i] = UINT32_MAX;
      }
    }
    int res;
    thrd_join(streamer->worker_thread, &res);
  }

  // make sure the GPU is idle before freeing resources
  cudaDeviceSynchronize();

  if (streamer->buffers[0])
    cudaFree(streamer->buffers[0]);

  for (u32 i = 0; i < 2; ++i) {
    if (streamer->upload_done[i])
      cudaFreeHost((void *)(uintptr_t)(streamer->upload_done[i]));
    if (streamer->gpu_done[i])
      cudaFreeHost((void *)(uintptr_t)(streamer->gpu_done[i]));
  }

  munmap_file(&streamer->mapped_file);
  free(streamer);
}

/* ---- acquire: make the compute stream wait for the upload ---------- */

float3 *
streamer_acquire_next_frame(Streamer const *const streamer, u32 const buf_idx, cudaStream_t const compute_stream) {
  if (streamer->total_frames == 0) {
    // don't wait if there are no frames
    return streamer->buffers[buf_idx];
  }

  u32 const wait_val = streamer->consumer_seq[buf_idx];

  cuStreamWaitValue32((CUstream)compute_stream, streamer->upload_done_dptr[buf_idx], wait_val,
                      CU_STREAM_WAIT_VALUE_GEQ);

  return streamer->buffers[buf_idx];
}

/* ---- release: tell the worker the buffer is free ------------------- */

void streamer_release_frame(Streamer *const streamer, u32 const buf_idx, cudaStream_t const compute_stream) {
  u32 const seq = streamer->consumer_seq[buf_idx];

  cuStreamWriteValue32((CUstream)compute_stream, (CUdeviceptr)streamer->gpu_done_dptr[buf_idx], seq,
                       CU_STREAM_WRITE_VALUE_DEFAULT);

  ++streamer->consumer_seq[buf_idx];
}
