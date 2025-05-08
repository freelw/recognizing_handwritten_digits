#include "kernel.cuh"

#ifndef GCC_ASAN

__global__ void fill_float(
    float *Md, int M, float value
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        Md[idx] = value;
    }
}

__global__ void tensor_add_2d(
    float *Md, float *Nd, float *Pd, //Pd is result
    int M, int N, // shape
    int stride_M0, int stride_M1,
    int stride_N0, int stride_N1,
    int stride_P0, int stride_P1
    ) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    } else {
        int index_M = row * stride_M0 + col * stride_M1;
        int index_N = row * stride_N0 + col * stride_N1;
        int index_P = row * stride_P0 + col * stride_P1;
        Pd[index_P] = Md[index_M] + Nd[index_N];
    }
}
#endif // GCC_ASAN