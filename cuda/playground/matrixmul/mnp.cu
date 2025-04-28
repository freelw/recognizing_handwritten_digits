#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define WIDTH 1024
#define TILE_WIDTH 32

const int M = 1024;
const int N = 777;
const int P = 500;

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

void naive_matrixmul(float *Md, float *Nd, float *Pd, int M, int N, int P) {
    // Naive matrix multiplication
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            Pd[i * P + j] = 0;
            for (int k = 0; k < N; ++k) {
                Pd[i * P + j] += Md[i * N + k] * Nd[k * P + j];
            }
        }
    }
}

bool check(float *h_output, float *res, int size) {
    for (int i = 0; i < size; ++i) {
        if (fabs(h_output[i] - res[i]) > 1e-3) {
            std::cout << "Error: " << "[" << i << "] " << h_output[i] << " != " << res[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    
    float *h_Md = (float *)malloc(M*N*sizeof(float));
    float *h_Nd = (float *)malloc(N*P*sizeof(float));
    float *h_Pd = (float *)malloc(M*P*sizeof(float));
    float *h_res = (float *)malloc(M*P*sizeof(float));
    float *d_Md;
    float *d_Nd;
    float *d_Pd;

    cudaMalloc((void **)&d_Md, M*N*sizeof(float));
    cudaMalloc((void **)&d_Nd, N*P*sizeof(float));
    cudaMalloc((void **)&d_Pd, M*P*sizeof(float));

    for (int i = 0; i < M*N; i++) {
        // h_Md[i] = 2 * (float)drand48() - 1.0;
        h_Md[i] = 1;
    }
    for (int i = 0; i < N*P; i++) {
        // h_Nd[i] = 2 * (float)drand48() - 1.0;
        h_Nd[i] = 1;
    }
    naive_matrixmul(h_Md, h_Nd, h_res, M, N, P);
    
    dim3 gridDim(
        (P + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH
    );
    dim3 blockDim(
        TILE_WIDTH,
        TILE_WIDTH
    );

    std::cout << "gridDim : " << gridDim.x << " " << gridDim.y << std::endl;
    std::cout << "blockDim : " << blockDim.x << " " << blockDim.y << std::endl;

    cudaMemcpy(d_Md, h_Md, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Nd, h_Nd, N*P*sizeof(float), cudaMemcpyHostToDevice);
    matrixmul<<<gridDim, blockDim>>>(d_Md, d_Nd, d_Pd, M, N, P);
    cudaMemcpy(h_Pd, d_Pd, M*P*sizeof(float), cudaMemcpyDeviceToHost);

    if (check(h_Pd, h_res, M*P)) {
        std::cout << "Success!" << std::endl;
    } else {
        std::cout << "Failed!" << std::endl;
    }
    free(h_Md);
    free(h_Nd);
    free(h_Pd);
    cudaFree(d_Md);
    cudaFree(d_Nd);
    cudaFree(d_Pd);
    return 0;
}