#include "kernel.cuh"

__global__ void matrixmul(
    float *Md, float *Nd, float *Pd, int M, int N, int P) {
    
    // printf("blockIdx.x %d, blockIdx.y %d, threadIdx.x %d, threadIdx.y %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("row %d, col %d\n", row, col);

    __shared__ float s_Md[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_Nd[TILE_WIDTH][TILE_WIDTH];

    
    for (int m = 0; m < (N + TILE_WIDTH - 1)/ TILE_WIDTH; ++m) {
        // Load data into shared memory
        s_Md[threadIdx.y][threadIdx.x] = row < M && m * TILE_WIDTH + threadIdx.x < N ? Md[row * N + m * TILE_WIDTH + threadIdx.x] : 0;
        // printf(
        //     "blockIdx.x %d, blockIdx.y %d, threadIdx.x %d, threadIdx.y %d\n"
        //     "s_Md[%d][%d] = %f\n"
        //     "m = %d, row = %d, N = %d, threadIdx.x = %d\n", 
        //     blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
        //     threadIdx.y, threadIdx.x, s_Md[threadIdx.y][threadIdx.x],
        //     m, row, N, threadIdx.x);
        s_Nd[threadIdx.y][threadIdx.x] = col < P && m * TILE_WIDTH + threadIdx.y < N ? Nd[(m * TILE_WIDTH + threadIdx.y) * P + col] : 0;
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