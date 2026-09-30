#include "streamer.h"
#include <stdlib.h>

static int worker_loop(void *arg) {
  AsyncFluidStreamer *streamer = (AsyncFluidStreamer *)arg;
  size_t current_frame = 0;
  int buf_idx = 0;

  while (!atomic_load(&streamer->stop_requested)) {
    // wait until target buffer is free to receive new frame
    mtx_lock(&streamer->mutex);
    while (streamer->buffer_ready[buf_idx] && !atomic_load(&streamer->stop_requested)) {
      cnd_wait(&streamer->cv_produce, &streamer->mutex);
    }
    mtx_unlock(&streamer->mutex);

    if (atomic_load(&streamer->stop_requested))
      break;

    // 1. file seek & read directly into pinned memory
    long offset = (long)(streamer->data_offset + current_frame * streamer->frame_bytes);
    if (fseek(streamer->file_stream, offset, SEEK_SET) != 0 ||
        fread(streamer->h_pinned[buf_idx], 1, streamer->frame_bytes, streamer->file_stream) != streamer->frame_bytes) {
      atomic_store(&streamer->stop_requested, true);
      break;
    }

    // 2. asynchronous H2D Transfer on transfer stream
    cudaMemcpyAsync(streamer->d_buffers[buf_idx], streamer->h_pinned[buf_idx], streamer->frame_bytes,
                    cudaMemcpyHostToDevice, streamer->transfer_stream);

    // 3. record hardware completion event
    cudaEventRecord(streamer->frame_ready_events[buf_idx], streamer->transfer_stream);

    // update status and notify consumer
    mtx_lock(&streamer->mutex);
    streamer->buffer_ready[buf_idx] = true;
    streamer->frame_loaded[buf_idx] = current_frame;
    mtx_unlock(&streamer->mutex);

    cnd_signal(&streamer->cv_consume);

    // advance indices
    current_frame = (current_frame + 1) % streamer->total_frames;
    buf_idx = 1 - buf_idx;
  }
  return 0;
}

AsyncFluidStreamer *streamer_create(char const *filepath) {
  AsyncFluidStreamer *streamer = (AsyncFluidStreamer *)malloc(sizeof(AsyncFluidStreamer));
  if (!streamer)
    return NULL;

  streamer->file_stream = fopen(filepath, "rb");
  if (!streamer->file_stream) {
    free(streamer);
    return NULL;
  }

  uint32_t header[2];
  if (fread(header, sizeof(header[0]), 2, streamer->file_stream) != 2 || header[0] == 0 || header[1] == 0) {
    fclose(streamer->file_stream);
    free(streamer);
    return NULL;
  }

  streamer->total_frames = header[0];
  streamer->num_particles = header[1];
  streamer->frame_bytes = streamer->num_particles * sizeof(float3);
  streamer->data_offset = sizeof(header);

  // CUDA streams & events
  cudaStreamCreate(&streamer->transfer_stream);
  cudaStreamCreate(&streamer->compute_stream);

  for (int i = 0; i < 2; ++i) {
    cudaMallocHost((void **)&streamer->h_pinned[i], streamer->frame_bytes);
    cudaMalloc((void **)&streamer->d_buffers[i], streamer->frame_bytes);
    cudaEventCreate(&streamer->frame_ready_events[i]);
    streamer->buffer_ready[i] = false;
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

float4 *streamer_acquire_next_frame(AsyncFluidStreamer *streamer, int buf_idx, cudaStream_t *out_compute_stream) {
  mtx_lock(&streamer->mutex);
  while (!streamer->buffer_ready[buf_idx]) {
    cnd_wait(&streamer->cv_consume, &streamer->mutex);
  }
  mtx_unlock(&streamer->mutex);

  // hardware-level stream dependency: compute stream waits for transfer stream
  cudaStreamWaitEvent(streamer->compute_stream, streamer->frame_ready_events[buf_idx], 0);

  *out_compute_stream = streamer->compute_stream;
  return streamer->d_buffers[buf_idx];
}

void streamer_release_frame(AsyncFluidStreamer *streamer, int buf_idx) {
  mtx_lock(&streamer->mutex);
  streamer->buffer_ready[buf_idx] = false;
  mtx_unlock(&streamer->mutex);

  cnd_signal(&streamer->cv_produce);
}

void streamer_destroy(AsyncFluidStreamer *streamer) {
  if (!streamer)
    return;

  atomic_store(&streamer->stop_requested, true);

  mtx_lock(&streamer->mutex);
  cnd_broadcast(&streamer->cv_produce);
  mtx_unlock(&streamer->mutex);

  int res;
  thrd_join(streamer->worker_thread, &res);

  for (int i = 0; i < 2; ++i) {
    cudaFreeHost(streamer->h_pinned[i]);
    cudaFree(streamer->d_buffers[i]);
    cudaEventDestroy(streamer->frame_ready_events[i]);
  }

  cudaStreamDestroy(streamer->transfer_stream);
  cudaStreamDestroy(streamer->compute_stream);

  mtx_destroy(&streamer->mutex);
  cnd_destroy(&streamer->cv_produce);
  cnd_destroy(&streamer->cv_consume);

  fclose(streamer->file_stream);
  free(streamer);
}
