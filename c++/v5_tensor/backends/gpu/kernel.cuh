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

__global__ void cross_entropy(
    float *Md, int32_t *labels,
    float *maxs, float *sums,
    float *loss,
    int M, int N,
    int stride0, int stride1
);

__global__ void cross_entropy_backward(
    float *Md, int32_t *labels,
    float *maxs, float *sums,
    float *grad,
    int M, int N,
    int Md_stride0, int Md_stride1,
    int grad_stride0, int grad_stride1
);

__global__ void tensor_relu(
    float *Md, float *Nd, int M
);

__global__ void tensor_relu_prime(
    float *Md, float *Nd, int M
);

__global__ void tensor_l2_norm(
    float *Md, float *Nd, int M
);

__global__ void tensor_clip(
    float *Md, float *Norm, int M,
    float clip_value
);

__global__ void tensor_adam_step(
    float *w, float *grad,
    float *m, float *v,
    int M, int t,
    float beta1, float beta2,
    float lr, float eps
);

__global__ void reshape_deep_cp_float_kernel(
    float *dst, float *src,
    int32_t *src_shape, int32_t *src_strides,
    int32_t dim, int32_t length
);

__global__ void repeat_interleave_int32_kernel(
    int32_t *src, int32_t *dst,
    int32_t src_length, int32_t dst_length,
    int32_t n
);

#endif // GCC_ASAN

#endif // V5_TENSOR_BACKENDS_GPU_KERNEL_CUH