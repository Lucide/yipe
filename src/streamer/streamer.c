#include "streamer.h"
#include <stdio.h>
#include <stdlib.h>

static int worker_loop(void *arg) { return 0; }

AsyncFluidStreamer *streamer_create(FILE *const file) {
  AsyncFluidStreamer *streamer = (AsyncFluidStreamer *)malloc(sizeof(AsyncFluidStreamer));

  MappedFile mapped_file;
  mmap_file(file, &mapped_file);

  if (mapped_file.size < sizeof(StreamHeader)) {
    munmap_file(&mapped_file);
    free(streamer);
    return NULL;
  }
  StreamHeader const header = ((Stream const *)mapped_file.data)->header;

  streamer->mapped_file = mapped_file;
  streamer->header = header;
  streamer->file_frame_bytes = header.num_particles * sizeof(float3);
  streamer->frame_bytes = header.num_particles * sizeof(float4);
  streamer->data_offset = sizeof(header);

  if (mapped_file.size < sizeof(header) + header.total_frames * streamer->file_frame_bytes) {
    munmap_file(&mapped_file);
    free(streamer);
    return NULL;
  }

  // CUDA streams & events
  cudaStreamCreate(&streamer->transfer_stream);

  for (u32 i = 0; i < 2; ++i) {
    cudaMalloc((void **)&streamer->d_buffers[i], streamer->frame_bytes);
    cudaEventCreate(&streamer->frame_ready_events[i]);
    cudaEventCreate(&streamer->frame_done_events[i]);
    streamer->buffer_ready[i] = false;
    streamer->frame_done_recorded[i] = false;
    streamer->frame_loaded[i] = 0;
  }

  // C11 sync primitive initializations
  atomic_init(&streamer->stop_requested, false);
  mtx_init(&streamer->mutex, mtx_plain);
  cnd_init(&streamer->cv_produce);
  cnd_init(&streamer->cv_consume);

  // launch background worker thread
  if (thrd_create(&streamer->worker_thread, worker_loop, streamer) != thrd_success) {
    streamer_destroy(streamer);
    return NULL;
  }

  return streamer;
}

float4 *streamer_acquire_next_frame(AsyncFluidStreamer *streamer, u32 buf_idx, cudaStream_t compute_stream) {}

void streamer_release_frame(AsyncFluidStreamer *streamer, u32 buf_idx, cudaStream_t compute_stream) {}

void streamer_destroy(AsyncFluidStreamer *streamer) {}
