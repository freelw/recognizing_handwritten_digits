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
    float *input,
    uint *labels,
    float *loss,
    int N, int C) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp_loss = 0;
    if (index >= N) {
        return ;
    } else {
        uint label = labels[index];
        float max_val = -1e10;
        
        for (int i = 0; i < C; ++i) {
            float z = input[i*C + index];
            max_val = fmaxf(max_val, z);
        }
        float sum = 0;
        for (int i = 0; i < C; ++i) {
            float z = input[i*C + index];
            sum += expf(z - max_val);
        }
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