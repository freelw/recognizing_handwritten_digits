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
#endif // GCC_ASAN

#endif // V5_TENSOR_BACKENDS_GPU_KERNEL_CUH