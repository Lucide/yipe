#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"
#include <cuda_runtime.h>

int main(int const argc, char const *const *const argv) {
  int const size = 5;
  float data[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  float *d_data;

  // Allocate and copy data to GPU
  cudaMalloc((void**)&d_data, size * sizeof(float));
  cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);

  // Launch the kernel (unsupported by standard C compilers)
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  launch_kernel1(blocksPerGrid, threadsPerBlock, d_data, size);

  // Copy result back and clean up
  cudaMemcpy(data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_data);

  for(int i = 0; i < size; i++) {
    printf("%f ", data[i]); // Outputs: 2.0 4.0 6.0 8.0 10.0
  }
  return 0;
}