#ifndef KERNEL_CUH
#define KERNEL_CUH


#include <cuda.h>
#include <cuda_runtime.h>

const int TILE_WIDTH = 32;

__global__ void matrixmul(float *Md, float *Nd, float *Pd, int M, int N, int P);
__global__ void expand_add_kernel(float *Md, float *Nd, int M, int N);
__global__ void relu_kernel(float *Md, int M);
__global__ void add_eq_kernel(float *Md, float *Nd, int M, int N);
__global__ void cross_entropy_loss(
    float *input, uint *labels, float *loss,
    float *maxs, float *sums,
    int N, int C);
__global__ void cross_entropy_loss_backward(
    float *w, float *grad, uint *labels,
    float *maxs, float *sums,
    int N, int C);
__global__ void sum(float *Md, float *Nd, int M, int N);
__global__ void transpose(float *Md, float *Nd, int M, int N);
__global__ void relu_prime(float *Md, int M);
__global__ void step_kernel(float lr, int t, float *w, float *grad, float *mm, float *mv, int M);

#endif