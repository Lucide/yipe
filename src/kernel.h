#ifndef YIPE_KERNEL_H
#define YIPE_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

// This is the clean C function your main.c file will call
void launch_kernel1(int blocksPerGrid, int threadsPerBlock, float *d_data,
                    int size);

#ifdef __cplusplus
}
#endif

#endif

