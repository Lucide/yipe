#include "kernel.h"
// #include <cuda_runtime.h>

// The actual GPU kernel
__global__ void kernel1(float *data, int const size) {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    data[id] *= 2.0f; // example operation
  }
}

// The wrapper function declared in kernel.h
extern "C" void launch_kernel1(int blocksPerGrid, int threadsPerBlock,
                               float *d_data, int const size) {
  kernel1<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
}
