#include "streamer.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdatomic.h>
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

// WORKER THREAD

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
    while (atomic_load_explicit(streamer->gpu_done[slot], memory_order_acquire) < seq &&
           !atomic_load(&streamer->stop_requested)) {
      thrd_yield();
    }

    if (atomic_load_explicit(&streamer->stop_requested, memory_order_acquire))
      return 0;

    // synchronoud copy
    TRAP_CUDAPI(cudaMemcpy(streamer->buffers[slot],
                           &((float3 const *)streamer->mapped_file.data)[(u64)streamer->num_particles * (u64)frame],
                           streamer->frame_size, cudaMemcpyHostToDevice),
                "streamer: worker thread upload frame");

    // publishing must also happen in the cpu with mapped memory because the gpu can deadlock
    // publish upload_done
    atomic_store_explicit(streamer->upload_done[slot], seq + 1, memory_order_release);
  }
  return 0;
}

// HELPERS

// allocate one pinned+mapped uint32 flag
static void alloc_flag(_Atomic u32 volatile **const host_ptr, CUdeviceptr *const dev_ptr, u32 const initial_value) {
  // allocate memory mapped flag (the cpu must be able to read and update it)
  void *hostptr;
  TRAP_CUDAPI(cudaHostAlloc(&hostptr, sizeof(_Atomic u32), cudaHostAllocMapped | cudaHostAllocPortable),
              "streamer: allocate mapped flag");

  atomic_init((_Atomic u32 *)hostptr, initial_value);

  // get device pointers for the mapped memory
  // the gpu will use this to wait/write
  CUdeviceptr deviceptr;
  TRAP_CUDAPI(cudaHostGetDevicePointer((void **)&deviceptr, hostptr, 0), "streamer: get device pointer to mapped flag");

  // pass back the pointers
  *host_ptr = hostptr;
  *dev_ptr = deviceptr;
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

  printf("File stats:\n"
         "\tnum particles: %u\n"
         "\tfile total_frames: %u\n"
         "\ttotal rendered frames:%u\n"
         "\tfile size: %lld\n"
         "\tframe size: %zu\n",
         header->num_particles, header->total_frames, total_frames, size, header->num_particles * sizeof(float3));

  header->total_frames = total_frames;

  return true;
}

// special error handler for low level driver errors
// only this file uses low level driver operations
static void handle_cuda_driver_error(CUresult const result, char const *message) {
  if (result != CUDA_SUCCESS) {
    char const *errorName = NULL;
    char const *errorStr = NULL;

    // Fetch driver error details
    cuGetErrorName(result, &errorName);
    cuGetErrorString(result, &errorStr);

    printf("[%s] CUDA error \"%s\": %s\n", message, errorName, errorStr);

    exit(-1);
  }
}

Streamer *streamer_create(FILE *const file, u32 const start_frame, u32 const total_frames, u8 const frameskip) {
  Streamer *streamer = malloc(sizeof(Streamer));

  StreamHeader header;
  if (!parse_header(&header, file, start_frame, total_frames, frameskip))
    return NULL;
  size_t const frame_size = header.num_particles * sizeof(float3);
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

  MmapError const err = mmap_file(file, map_start, map_size, &streamer->mapped_file);
  if (err != MMAP_SUCCESS) {
    printf("mmap error: %d\n", err);
    exit(-err);
  }

  // allocate two device buffers (one contiguous allocation)
  float3 *buffers;
  TRAP_CUDAPI(cudaMalloc((void **)&buffers, 2 * frame_size), "streamer: allocate device buffers");
  streamer->buffers[0] = buffers;
  streamer->buffers[1] = &buffers[header.num_particles];

  // allocate pinned+mapped synchronisation flags
  for (u32 i = 0; i < 2; ++i) {
    streamer->consumer_seq[i] = 1;
    alloc_flag(&streamer->upload_done[i], &streamer->upload_done_dptr[i], 0);
    alloc_flag(&streamer->gpu_done[i], &streamer->gpu_done_dptr[i], 0);
  }

  atomic_init(&streamer->stop_requested, false);

  // launch background worker thread
  if (thrd_create(&streamer->worker_thread, worker_loop, streamer) != thrd_success) {
    atomic_store_explicit(&streamer->stop_requested, true, memory_order_release);
    streamer_destroy(streamer);
    return NULL;
  }

  return streamer;
}

void streamer_destroy(Streamer *const streamer) {
  if (!streamer)
    return;

  // signal the worker to stop and join it
  if (!atomic_load_explicit(&streamer->stop_requested, memory_order_acquire)) {
    atomic_store_explicit(&streamer->stop_requested, true, memory_order_release);
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
  TRAP_CUDAPI(cudaDeviceSynchronize(), "streamer: synchronize while destroying streamer");

  if (streamer->buffers[0])
    TRAP_CUDAPI(cudaFree(streamer->buffers[0]), "streamer: free device buffers");

  for (u32 i = 0; i < 2; ++i) {
    if (streamer->upload_done[i])
      TRAP_CUDAPI(cudaFreeHost((void *)(uintptr_t)(streamer->upload_done[i])), "streamer: free flag");
    if (streamer->gpu_done[i])
      TRAP_CUDAPI(cudaFreeHost((void *)(uintptr_t)(streamer->gpu_done[i])), "streamer: free flag");
  }

  munmap_file(&streamer->mapped_file);
  free(streamer);
}

// acquire: make the compute stream wait for the upload

float3 *
streamer_acquire_next_frame(Streamer const *const streamer, u32 const buf_idx, cudaStream_t const compute_stream) {
  u32 const wait_val = streamer->consumer_seq[buf_idx];

  handle_cuda_driver_error(
      cuStreamWaitValue32(compute_stream, streamer->upload_done_dptr[buf_idx], wait_val, CU_STREAM_WAIT_VALUE_GEQ),
      "streamer: wait to aquire frame");

  return streamer->buffers[buf_idx];
}

// release: tell the worker the buffer is free

void streamer_release_frame(Streamer *const streamer, u32 const buf_idx, cudaStream_t const compute_stream) {
  u32 const seq = streamer->consumer_seq[buf_idx];

  handle_cuda_driver_error(
      cuStreamWriteValue32(compute_stream, streamer->gpu_done_dptr[buf_idx], seq, CU_STREAM_WRITE_VALUE_DEFAULT),
      "streamer: release frame");

  ++streamer->consumer_seq[buf_idx];
}
