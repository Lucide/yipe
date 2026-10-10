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

/*
 * Double-buffered frame streamer
 * ==============================
 *
 * A dedicated worker thread performs synchronous cudaMemcpy from
 * memory-mapped host pages into one of two GPU buffers. Synchronisation
 * with the compute / render stream is done entirely with
 * cuStreamWaitValue32 / cuStreamWriteValue32 on host-pinned (device-
 * mapped) uint32 flags - no CUDA events or extra CUDA streams are needed.
 *
 * Two flags per buffer slot:
 *
 *   upload_done[i] – bumped by the CPU worker  AFTER the memcpy into
 *                    buffers[i] is complete.
 *                    The GPU compute stream waits on this before reading.
 *
 *   gpu_done[i]    – bumped by the GPU compute stream when it has
 *                    finished reading buffers[i].
 *                    The CPU worker spins on this before writing again.
 *
 * All counters are monotonically increasing sequence numbers to prevent
 * ABA issues: each frame upload uses a fresh, strictly larger value.
 *
 * Protocol
 * --------
 *   Worker thread (slot s, sequence N):
 *       1. spin-wait until *gpu_done[s] >= N          - buffer free
 *       2. cudaMemcpy(buffers[s], host_frame, ...)    - synchronous H→D
 *       3. *upload_done[s] = N+1                      - signal GPU
 *
 *   Compute stream (slot s, called by main thread):
 *       acquire:  cuStreamWaitValue32(stream, upload_done_dptr[s], N+1, GEQ)
 *       release:  cuStreamWriteValue32(stream, gpu_done_dptr[s], N+1, DEFAULT)
 *
 * Initial values: upload_done = 0, gpu_done = 1 (both slots "free").
 */

/* ------------------------------------------------------------------ */
/*  Worker thread                                                     */
/* ------------------------------------------------------------------ */

static int worker_loop(void const *const arg) {
  Streamer const *const streamer = (Streamer const *)arg;
  u8 slot = 0;
  u32 seq = 0;

  for (u32 frame = 0; !atomic_load(&streamer->stop_requested);
       frame = (frame + streamer->frameskip + 1) % streamer->total_frames) {
    /*
     * Sequence number for this particular upload into `slot`.
     * slot 0 is used by frames 0, 2, 4, … → seq 1, 2, 3, …
     * slot 1 is used by frames 1, 3, 5, … → seq 1, 2, 3, …
     */

    /* 1. Wait until the GPU is done with this slot.
     *    gpu_done[slot] is initialised to 1 and bumped to `seq` by the
     *    GPU after it finishes reading the *previous* upload in this
     *    slot.  For the very first frame in each slot (seq == 1) the
     *    initial value 1 satisfies the wait immediately.
     */
    // printf("producer waiting for %u on slot %u\n", seq, slot);
    while (*streamer->gpu_done[slot] < seq && !atomic_load(&streamer->stop_requested)) {
      thrd_yield();
    }

    if (atomic_load(&streamer->stop_requested))
      return 0;

    /* 2. Synchronous host-to-device copy (fine on a worker thread). */
    cudaMemcpy(streamer->buffers[slot], &((float3 const *)streamer->mapped_file.data)[streamer->num_particles * frame],
               streamer->frame_size, cudaMemcpyHostToDevice);

    /* 3. Tell the GPU the upload is ready (visible via device pointer). */
    // printf("producer publishing %u on slot %u\n", seq + 1, slot);
    *streamer->upload_done[slot] = seq + 1;

    // increment once every two frame loaded
    seq += slot;
    slot = (slot + 1) % 2;
  }
  return 0;
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

// allocate one pinned+mapped uint32 flag
static int alloc_flag(u32 volatile **const host_ptr, unsigned long long *const dev_ptr, u32 const initial_value) {
  /*
   * cudaHostAlloc with cudaHostAllocMapped gives us:
   *   - a host pointer the CPU can read/write directly
   *   - a device pointer usable with cuStreamWaitValue32 et al.
   */
  u32 *h = NULL;
  if (cudaHostAlloc((void **)&h, sizeof(u32), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess)
    return -1;

  *h = initial_value;

  /*
   * cudaHostGetDevicePointer returns a device-side alias.
   * cuStreamWaitValue / cuStreamWriteValue require a CUdeviceptr,
   * which is just a typedef for unsigned long long.
   */
  void *d = NULL;
  if (cudaHostGetDevicePointer(&d, (void *)h, 0) != cudaSuccess) {
    cudaFreeHost((void *)h);
    return -1;
  }

  *host_ptr = (u32 volatile *)h;
  *dev_ptr = (unsigned long long)(uintptr_t)d;
  return 0;
}

static bool parse_header(
    StreamHeader *const header, FILE *const file, u32 const start_frame, u32 const total_frames, u8 const frameskip) {
  size_t const bytes_read = fread(header, 1, sizeof(StreamHeader), file);
  if (bytes_read < sizeof(StreamHeader)) {
    if (ferror(file)) {
      perror("error reading recording header");
      return false;
    }
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

  if (header->total_frames == 0 || header->num_particles == 0)
    return false;

  return true;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                        */
/* ------------------------------------------------------------------ */

Streamer *streamer_create(FILE *const file, u32 const start_frame, u32 const total_frames, u8 const frameskip) {
  Streamer *streamer = malloc(sizeof(Streamer));

  StreamHeader header;
  parse_header(&header, file, start_frame, total_frames, frameskip);
  size_t const frame_size = header.num_particles * sizeof(float3);
  size_t const map_start = sizeof(StreamHeader) + start_frame * frame_size;
  size_t const map_size = total_frames * frame_size;

  *streamer = (Streamer){
      .room = header.room,
      .total_frames = total_frames,
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

  /* Allocate pinned+mapped synchronisation flags.
   *
   * upload_done[i] starts at 0  →  nothing uploaded yet.
   * gpu_done[i]    starts at 1  →  both slots are "free" so the worker
   *                                 can begin immediately.
   */
  for (u32 i = 0; i < 2; ++i) {
    streamer->consumer_seq[i] = 1;
    if (alloc_flag(&streamer->upload_done[i], &streamer->upload_done_dptr[i], 0) != 0 ||
        alloc_flag(&streamer->gpu_done[i], &streamer->gpu_done_dptr[i], 0) != 0) {
      streamer_destroy(streamer);
      return NULL;
    }
  }

  cudaStreamCreate(&streamer->worker_stream);
  atomic_init(&streamer->stop_requested, false);

  /* launch background worker thread. */
  if (thrd_create(&streamer->worker_thread, (thrd_start_t)worker_loop, streamer) != thrd_success) {
    atomic_store(&streamer->stop_requested, true);
    streamer_destroy(streamer);
    return NULL;
  }

  return streamer;
}

void streamer_destroy(Streamer *const streamer) {
  if (!streamer)
    return;

  /* Signal the worker to stop and join it. */
  if (!atomic_load(&streamer->stop_requested)) {
    atomic_store(&streamer->stop_requested, true);
    /* Unblock the worker thread if it's waiting on cudaStreamSynchronize */
    for (u32 i = 0; i < 2; ++i) {
      if (streamer->gpu_done[i]) {
        *streamer->gpu_done[i] = UINT32_MAX;
      }
    }
    int res;
    thrd_join(streamer->worker_thread, &res);
  }

  if (streamer->worker_stream) {
    cudaStreamDestroy(streamer->worker_stream);
  }

  /* Make sure the GPU is idle before freeing resources. */
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
  /*
   * The worker thread writes *upload_done[buf_idx] = seq when the
   * memcpy is done.  We need to know the current expected sequence
   * number.  The latest value already stored tells us what to wait for:
   *
   *   current value V  →  we are waiting for V+1  (next upload).
   *
   * But because the worker may have *just* written V and the compute
   * stream hasn't released V-1 yet, the simplest correct approach is to
   * wait for a value strictly greater than what we last consumed.
   *
   * However, because we pair acquire/release 1:1 in the render loop and
   * the worker increments upload_done monotonically, we can simply read
   * the current host value and wait for (value+1) - the NEXT upload.
   *
   * In practice, the very first time buf_idx 0 is acquired the stored
   * value is 0, so we wait for >= 1, which is the first upload.
   */

  if (streamer->total_frames == 0) {
    // don't wait if there are no frames
    return streamer->buffers[buf_idx];
  }

  u32 const wait_val = streamer->consumer_seq[buf_idx];

  // printf("consumer waiting for %u on slot %u\n", wait_val, buf_idx);
  cuStreamWaitValue32((CUstream)compute_stream, (CUdeviceptr)streamer->upload_done_dptr[buf_idx], wait_val,
                      CU_STREAM_WAIT_VALUE_GEQ);

  return streamer->buffers[buf_idx];
}

/* ---- release: tell the worker the buffer is free ------------------- */

void streamer_release_frame(Streamer *const streamer, u32 const buf_idx, cudaStream_t const compute_stream) {
  /*
   * We write the *same* sequence number that the worker wrote into
   * upload_done[buf_idx] - the worker waits for gpu_done >= seq before
   * reusing this slot.
   *
   * upload_done was updated to `seq` by the worker; at this point the
   * value is already visible on the host so we can read it directly.
   */
  u32 const seq = streamer->consumer_seq[buf_idx];

  // printf("consumer publishing %u on slot %u\n", seq, buf_idx);
  cuStreamWriteValue32((CUstream)compute_stream, (CUdeviceptr)streamer->gpu_done_dptr[buf_idx], seq,
                       CU_STREAM_WRITE_VALUE_DEFAULT);

  // The consumer is done with this sequence, expect the next one
  ++streamer->consumer_seq[buf_idx];
}
