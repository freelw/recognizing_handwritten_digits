#ifndef V5_TENSOR_BACKENDS_GPU_KERNEL_CUH
#define V5_TENSOR_BACKENDS_GPU_KERNEL_CUH

#ifndef GCC_ASAN
#include <cuda.h>
#include <cuda_runtime.h>

const int TILE_WIDTH = 32;

__global__ void fill_float(
    float *Md, int M, float value
);

__global__ void tensor_add_2d(
    float *Md, float *Nd, float *Pd, //Pd is result
    int M, int N, // shape
    int stride_M0, int stride_M1,
    int stride_N0, int stride_N1,
    int stride_P0, int stride_P1
);

__global__ void tensor_at_2d(
    float *Md, float *Nd, float *Pd,
    int M, int N, int P,
    int stride_M0, int stride_M1,
    int stride_N0, int stride_N1,
    int stride_P0, int stride_P1
);

__global__ void tensor_add_eq_1d(
    float *Md, float *Nd, int M
);

__global__ void tensor_add_eq_2d(
    float *Md, float *Nd,
    int M, int N,
    int stride_M0, int stride_M1,
    int stride_N0, int stride_N1
);

__global__ void expand_add(
    float *Md, float *Nd, float *Pd,
    int M, int N,
    int stride_M0, int stride_M1,
    int stride_P0, int stride_P1
);

__global__ void tensor_mul_2d(
    float *Md, float *Nd, float *Pd,
    int M, int N,
    int stride_M0, int stride_M1,
    int stride_N0, int stride_N1,
    int stride_P0, int stride_P1
);

__global__ void tensor_sum_2d_dim0(
    float *Md, float *Pd,
    int M, int N,
    int stride_M0, int stride_M1
);

#endif // GCC_ASAN

#endif // V5_TENSOR_BACKENDS_GPU_KERNEL_CUH