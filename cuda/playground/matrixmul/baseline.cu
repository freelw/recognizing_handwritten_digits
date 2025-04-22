#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

#define TILE_WIDTH 32

__global__ void matrixmul(
    float *Md, float *Nd, float *Pd, int width) {
    //naive
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    for (int k = 0; k < width; ++k) {
        sum += Md[row * width + k] * Nd[k * width + col];
    }
    Pd[row * width + col] = sum;
}

void naive_matrixmul(float *Md, float *Nd, float *Pd, int width) {
    // Naive matrix multiplication
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            Pd[i * width + j] = 0;
            for (int k = 0; k < width; ++k) {
                Pd[i * width + j] += Md[i * width + k] * Nd[k * width + j];
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
    const int WIDTH = 1024;
    float *h_Md = (float *)malloc(WIDTH*WIDTH*sizeof(float));
    float *h_Nd = (float *)malloc(WIDTH*WIDTH*sizeof(float));
    float *h_Pd = (float *)malloc(WIDTH*WIDTH*sizeof(float));
    float *h_res = (float *)malloc(WIDTH*WIDTH*sizeof(float));
    float *d_Md;
    float *d_Nd;
    float *d_Pd;

    cudaMalloc((void **)&d_Md, WIDTH*WIDTH*sizeof(float));
    cudaMalloc((void **)&d_Nd, WIDTH*WIDTH*sizeof(float));
    cudaMalloc((void **)&d_Pd, WIDTH*WIDTH*sizeof(float));

    for (int i = 0; i < WIDTH*WIDTH; i++) {
        h_Md[i] = 2 * (float)drand48() - 1.0;
        h_Nd[i] = 2 * (float)drand48() - 1.0;
    }
    naive_matrixmul(h_Md, h_Nd, h_res, WIDTH);
    
    dim3 gridDim(
        (WIDTH + TILE_WIDTH - 1) / TILE_WIDTH,
        (WIDTH + TILE_WIDTH - 1) / TILE_WIDTH
    );
    dim3 blockDim(
        TILE_WIDTH,
        TILE_WIDTH
    );

    cudaMemcpy(d_Md, h_Md, WIDTH*WIDTH*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Nd, h_Nd, WIDTH*WIDTH*sizeof(float), cudaMemcpyHostToDevice);
    matrixmul<<<gridDim, blockDim>>>(d_Md, d_Nd, d_Pd, WIDTH);
    cudaMemcpy(h_Pd, d_Pd, WIDTH*WIDTH*sizeof(float), cudaMemcpyDeviceToHost);

    if (check(h_Pd, h_res, WIDTH*WIDTH)) {
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