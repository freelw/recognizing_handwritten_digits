#include "kernel.cuh"

__global__ void matrixmul(
    float *Md, float *Nd, float *Pd, int M, int N, int P) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_Md[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_Nd[TILE_WIDTH][TILE_WIDTH];
    
    for (int m = 0; m < (N + TILE_WIDTH - 1)/ TILE_WIDTH; ++m) {
        // Load data into shared memory
        s_Md[threadIdx.y][threadIdx.x] = row < M && m * TILE_WIDTH + threadIdx.x < N ? Md[row * N + m * TILE_WIDTH + threadIdx.x] : 0.f;
        s_Nd[threadIdx.y][threadIdx.x] = col < P && m * TILE_WIDTH + threadIdx.y < N ? Nd[(m * TILE_WIDTH + threadIdx.y) * P + col] : 0.f;
        __syncthreads();
        if (row >= M || col >= P) {
            
        } else {
            float sum = 0;
            for (int k = 0; k < TILE_WIDTH; ++k) {
                sum += s_Md[threadIdx.y][k] * s_Nd[k][threadIdx.x];
            }
            Pd[row * P + col] += sum;
        }
        __syncthreads();
    }
}

__global__ void expand_add_kernel(
    float *Md, float *Nd, int M, int N) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
            
    } else {
       Md[row * N + col] += Nd[row];
    }
}

__global__ void relu_kernel(float *Md, int M) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= M) {
            
    } else {
        Md[index] = fmaxf(0, Md[index]);
    }
}

__global__ void add_eq_kernel(float *Md, float *Nd, int M, int N) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
            
    } else {
       Md[row * N + col] += Nd[row * N + col];
    }
}

__global__ void cross_entropy_loss(
    float *input, uint *labels, float *loss,
    float *maxs, float *sums,
    int N, int C) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp_loss = 0;
    if (index >= N) {
        return ;
    } else {
        uint label = labels[index];
        float max_val = -1e10;
        
        for (int i = 0; i < C; ++i) {
            float z = input[i*N + index];
            max_val = fmaxf(max_val, z);
        }
        maxs[index] = max_val;
        float sum = 0;
        for (int i = 0; i < C; ++i) {
            float z = input[i*N + index];
            sum += expf(z - max_val);
        }
        sums[index] = sum;
        float zt = input[label * N + index];
        tmp_loss = -zt + max_val + logf(sum);
    }
    atomicAdd(loss, tmp_loss);
    __syncthreads();
    // reduce avg tmp_loss
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *loss = (*loss) / N;
    }
}

__global__ void cross_entropy_loss_backward(
    float *w, float *grad, uint *labels,
    float *maxs, float *sums,
    int N, int C) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) {
        return ;
    } else {
        uint label = labels[index];
        float max_val = maxs[index];
        float sum = sums[index];
        for (int i = 0; i < C; ++i) {
            float z = w[i*N + index];
            float res = 0;
            if (i == label) {
                res = expf(z - max_val) / sum - 1;
            } else {
                res = expf(z - max_val) / sum;
            }
            res /= N;
            grad[i*N + index] = res;
        }
    }
}

__global__ void sum(float *Md, float *Nd, int M, int N) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= M) {
        return ;
    } else {
        float sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += Md[index * N + i];
        }
        Nd[index] = sum;
    }
}

__global__ void transpose(float *Md, float *Nd, int M, int N) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return ;
    } else {
        Nd[col * M + row] = Md[row * N + col];
    }
}

__global__ void relu_prime(float *Md, int M) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= M) {
        return ;
    } else {
        Md[index] = Md[index] > 0 ? 1 : 0;
    }
}

__global__ void step_kernel(
    float lr, int t, float *w, float *grad,
    float *mm, float *mv,
    int M) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= M) {
        return ;
    } else {
        const float beta1 = 0.9;
        const float beta2 = 0.999;
        const float eps = 1e-8;

        float g = grad[index];
        float _mm = mm[index];
        float _mv = mv[index];

        _mm = beta1 * _mm + (1 - beta1) * g;
        _mv = beta2 * _mv + (1 - beta2) * powf(g, 2);
        float mmt = _mm / (1 - powf(beta1, t));
        float mvt = _mv / (1 - powf(beta2, t));
        w[index] -= lr * mmt / (sqrtf(mvt) + eps);
        mm[index] = _mm;
        mv[index] = _mv;
    }
}