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

__global__ void tensor_at_2d(
    float *Md, float *Nd, float *Pd,
    int M, int N, int P,
    int stride_M0, int stride_M1,
    int stride_N0, int stride_N1,
    int stride_P0, int stride_P1
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_Md[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_Nd[TILE_WIDTH][TILE_WIDTH];
    
    for (int m = 0; m < (N + TILE_WIDTH - 1)/ TILE_WIDTH; ++m) {
        // Load data into shared memory

        int M_row = row;
        int M_col = m * TILE_WIDTH + threadIdx.x;
        int N_row = m * TILE_WIDTH + threadIdx.y;
        int N_col = col;
        s_Md[threadIdx.y][threadIdx.x] =
            M_row < M && M_col < N ?
            Md[M_row * stride_M0 + M_col * stride_M1] : 0.f;
        s_Nd[threadIdx.y][threadIdx.x] =
            N_row < N && N_col < P?
            Nd[N_row * stride_N0 + N_col * stride_N1] : 0.f;
        __syncthreads();
        if (row >= M || col >= P) {
            
        } else {
            float sum = 0.0f;
            for (int k = 0; k < TILE_WIDTH; ++k) {
                sum += s_Md[threadIdx.y][k] * s_Nd[k][threadIdx.x];
            }
            Pd[row * stride_P0 + col * stride_P1] += sum;
        }
        __syncthreads();
    }
}

__global__ void tensor_add_eq_1d(
    float *Md, float *Nd, int M
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= M) {
        return;
    } else {
        Md[index] += Nd[index];
    }
}

__global__ void tensor_add_eq_2d(
    float *Md, float *Nd,
    int M, int N,
    int stride_M0, int stride_M1,
    int stride_N0, int stride_N1
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    } else {
        int index_M = row * stride_M0 + col * stride_M1;
        int index_N = row * stride_N0 + col * stride_N1;
        Md[index_M] += Nd[index_N];
    }
}

__global__ void expand_add(
    float *Md, float *Nd, float *Pd,
    int M, int N,
    int stride_M0, int stride_M1,
    int stride_P0, int stride_P1
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    } else {
        int index_M = row * stride_M0 + col * stride_M1;
        int index_P = row * stride_P0 + col * stride_P1;
        Pd[index_P] = Md[index_M] + Nd[col];
    }
}

__global__ void tensor_mul_2d(
    float *Md, float *Nd, float *Pd,
    int M, int N,
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
        Pd[index_P] = Md[index_M] * Nd[index_N];
    }
}

__global__ void tensor_sum_2d_dim0(
    float *Md, float *Pd,
    int M, int N,
    int stride_M0, int stride_M1
) {
    // todo: this kernel should be optimized
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) {
        return;
    } else {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += Md[i * stride_M0 + col * stride_M1];
        }
        Pd[col] = sum;
    }
}

__global__ void cross_entropy(
    float *Md, int32_t *labels,
    float *maxs, float *sums,
    float *loss,
    int M, int N,
    int stride0, int stride1
) {
    extern __shared__ float partial_sums[];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    float tmp_loss = 0;
    partial_sums[tid] = 0.0f;
    if (row >= M) {
        return;
    } else {
        float max = -1e10;
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            float val = Md[row * stride0 + i * stride1];
            max = fmaxf(max, val);
        }
        maxs[row] = max;
        for (int i = 0; i < N; ++i) {
            float val = Md[row * stride0 + i * stride1];
            sum += expf(val - max);
        }
        sums[row] = sum;
        int32_t label = labels[row];
        float zt = Md[row * stride0 + label * stride1];
        tmp_loss = -zt + max + logf(sum);
    }
    partial_sums[tid] = tmp_loss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(loss, partial_sums[0]);
    }
}

__global__ void cross_entropy_backward(
    float *Md, int32_t *labels,
    float *maxs, float *sums,
    float *grad,
    int M, int N,
    int Md_stride0, int Md_stride1,
    int grad_stride0, int grad_stride1
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) {
        return;
    } else {
        float max = maxs[row];
        float sum = sums[row];
        int label = labels[row];
        for (int i = 0; i < N; ++i) {
            float val = Md[row * Md_stride0 + i * Md_stride1];
            grad[row * grad_stride0 + i * grad_stride1] = i == label ?
                (expf(val - max) / sum - 1) / M :
                expf(val - max) / sum / M ;
        }
    }
}

__global__ void tensor_relu(
    float *Md, float *Nd, int M
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= M) {
        return;
    } else {
        Nd[index] = fmaxf(Md[index], 0.f);
    }
}

__global__ void tensor_relu_prime(
    float *Md, float *Nd, int M
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= M) {
        return;
    } else {
        Nd[index] = Md[index] > 0.f ? 1.f : 0.f;
    }
}

__global__ void tensor_l2_norm(
    float *Md, float *Nd, int M
) {
    extern __shared__ float partial_sums[];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    partial_sums[tid] = 0.0f;

    if (row >= M) {
        return;
    } else {
        partial_sums[tid] = powf(Md[row], 2);
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(Nd, partial_sums[0]);
    }
}

__global__ void tensor_clip(
    float *Md, float *Norm, int M,
    float clip_value
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= M) {
        return;
    } else {
        float norm = sqrtf(Norm[0]);
        if (norm > clip_value) {
            Md[index] *= clip_value / norm;
        }
    }
}

__global__ void tensor_adam_step(
    float *w, float *grad,
    float *m, float *v, int M,
    float beta1, float beta2,
    float lr, float eps
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= M) {
        return;
    } else {
        float w_value = w[index];
        float m_value = m[index];
        float v_value = v[index];
        float grad_value = grad[index];

        m_value = beta1 * m_value + (1 - beta1) * grad_value;
        v_value = beta2 * v_value + (1 - beta2) * powf(grad_value, 2);
        float m_hat = m_value / (1 - powf(beta1, 1));
        float v_hat = v_value / (1 - powf(beta2, 1));
        w_value -= lr * m_hat / (sqrtf(v_hat) + eps);
        w[index] = w_value;
    }
}

#endif // GCC_ASAN