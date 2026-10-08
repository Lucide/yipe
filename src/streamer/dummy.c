#include "dummy.h"
#include <stdio.h>
#include <stdlib.h>

static u32 worker_loop(void *arg) {
  AsyncFluidStreamer *streamer = (AsyncFluidStreamer *)arg;
  u32 current_frame = 0;
  u32 buf_idx = 0;

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
    u32 const offset = streamer->data_offset + current_frame * streamer->file_frame_bytes;
    if (fseek(streamer->file_stream, (s32)offset, SEEK_SET) != 0 ||
        fread(streamer->h_file_buffers[buf_idx], 1, streamer->file_frame_bytes, streamer->file_stream) !=
            streamer->file_frame_bytes) {
      atomic_store(&streamer->stop_requested, true);
      break;
    }

    for (u32 i = 0; i < streamer->header.num_particles; ++i) {
      streamer->h_pinned[buf_idx][i].x = streamer->h_file_buffers[buf_idx][i].x;
      streamer->h_pinned[buf_idx][i].y = streamer->h_file_buffers[buf_idx][i].y;
      streamer->h_pinned[buf_idx][i].z = streamer->h_file_buffers[buf_idx][i].z;
      streamer->h_pinned[buf_idx][i].w = 0.0f;
    }

    if (streamer->frame_done_recorded[buf_idx]) {
      cudaStreamWaitEvent(streamer->transfer_stream, streamer->frame_done_events[buf_idx], 0);
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
    current_frame = (current_frame + 1) % streamer->header.total_frames;
    buf_idx = 1 - buf_idx;
  }
  return 0;
}

AsyncFluidStreamer *streamer_create(FILE *const file) {
  AsyncFluidStreamer *streamer = (AsyncFluidStreamer *)malloc(sizeof(AsyncFluidStreamer));
  if (!streamer)
    return NULL;

  // streamer->file_stream = fopen(filepath, "rb");
  // if (!streamer->file_stream) {
  //   free(streamer);
  //   return NULL;
  // }
  MappedFile mapped_file;
  mmap_file(file, &mapped_file);

  // StreamHeader header;
  // if (fread(&header, sizeof(header), 1, streamer->file_stream) != 1) {
  //   fclose(streamer->file_stream);
  //   free(streamer);
  //   return NULL;
  // }
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
    cudaMallocHost((void **)&streamer->h_pinned[i], streamer->frame_bytes);
    cudaMalloc((void **)&streamer->d_buffers[i], streamer->frame_bytes);
    cudaEventCreate(&streamer->frame_ready_events[i]);
    cudaEventCreate(&streamer->frame_done_events[i]);
    streamer->h_file_buffers[i] = (float3 *)malloc(streamer->file_frame_bytes);
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

float4 *streamer_acquire_next_frame(AsyncFluidStreamer *streamer, u32 buf_idx, cudaStream_t compute_stream) {
  mtx_lock(&streamer->mutex);
  while (!streamer->buffer_ready[buf_idx]) {
    cnd_wait(&streamer->cv_consume, &streamer->mutex);
  }
  mtx_unlock(&streamer->mutex);

  // hardware-level stream dependency: compute stream waits for transfer stream
  cudaStreamWaitEvent(compute_stream, streamer->frame_ready_events[buf_idx], 0);

  return streamer->d_buffers[buf_idx];
}

void streamer_release_frame(AsyncFluidStreamer *streamer, u32 buf_idx, cudaStream_t compute_stream) {
  cudaEventRecord(streamer->frame_done_events[buf_idx], compute_stream);

  mtx_lock(&streamer->mutex);
  streamer->buffer_ready[buf_idx] = false;
  streamer->frame_done_recorded[buf_idx] = true;
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

  u32 res;
  thrd_join(streamer->worker_thread, &res);
  cudaStreamSynchronize(streamer->transfer_stream);

  for (u32 i = 0; i < 2; ++i) {
    free(streamer->h_file_buffers[i]);
    cudaFreeHost(streamer->h_pinned[i]);
    cudaFree(streamer->d_buffers[i]);
    cudaEventDestroy(streamer->frame_ready_events[i]);
    cudaEventDestroy(streamer->frame_done_events[i]);
  }

  cudaStreamDestroy(streamer->transfer_stream);

  mtx_destroy(&streamer->mutex);
  cnd_destroy(&streamer->cv_produce);
  cnd_destroy(&streamer->cv_consume);

  fclose(streamer->file_stream);
  free(streamer);
}
