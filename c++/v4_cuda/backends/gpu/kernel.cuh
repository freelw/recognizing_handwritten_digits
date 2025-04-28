#ifndef KERNEL_CUH
#define KERNEL_CUH


#include <cuda.h>
#include <cuda_runtime.h>

const int TILE_WIDTH = 32;

__global__ void matrixmul(float *Md, float *Nd, float *Pd, int M, int N, int P);
__global__ void expand_add_kernel(float *Md, float *Nd, int M, int N);
__global__ void relu_kernel(float *Md, int M);

#endif