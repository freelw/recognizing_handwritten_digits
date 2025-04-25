#include "kernel.cuh"

__global__ void matrixmul(
    float *Md, float *Nd, float *Pd, int M, int N, int P) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float s_Md[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_Nd[TILE_WIDTH][TILE_WIDTH];

    float sum = 0;
    for (int m = 0; m < width / TILE_WIDTH; ++m) {
        // Load data into shared memory
        s_Md[threadIdx.y][threadIdx.x] = Md[row * width + m * TILE_WIDTH + threadIdx.x];
        s_Nd[threadIdx.y][threadIdx.x] = Nd[(m * TILE_WIDTH + threadIdx.y) * width + col];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += s_Md[threadIdx.y][k] * s_Nd[k][threadIdx.x];
        }
        __syncthreads();
    }
    Pd[row * width + col] = sum;
}