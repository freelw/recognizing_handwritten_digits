#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

#define THREAD_PER_BLOCK 256
#define EPOCHS 8

__global__ void reduce(float *d_input, float *d_output) {
    float *sdata = d_input + blockIdx.x * blockDim.x;
    __shared__ float share[THREAD_PER_BLOCK];
    share[threadIdx.x] = sdata[threadIdx.x];
    __syncthreads();
    int pow = 1;
    for (int i = 0; i < EPOCHS; ++i) {
        pow *= 2;
        if (threadIdx.x % pow == 0) {
            share[threadIdx.x] += share[threadIdx.x + pow / 2];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_output[blockIdx.x] = share[0];
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
    const int N = 32 * 1024 * 1024;
    const int size = N * sizeof(float);
    int block_num = N / THREAD_PER_BLOCK;
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(sizeof(float)*block_num);
    float *res = (float *)malloc(sizeof(float)*block_num);
    memset(res, 0, sizeof(float)*block_num);
    float *d_input;
    float *d_output;
    
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, sizeof(float)*block_num);
    
    for (int i = 0; i < N; i++) {
        h_input[i] = 2 * (float)drand48() - 1.0;
        // h_input[i] = 1;
    }

    for (int i = 0; i < block_num; ++ i) {
        for (int j = 0; j < THREAD_PER_BLOCK; ++ j) {
            res[i] += h_input[i * THREAD_PER_BLOCK + j];
        }
    }
    dim3 gridDim(block_num);
    dim3 blockDim(THREAD_PER_BLOCK);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    reduce<<<gridDim, blockDim>>>(d_input, d_output);
    cudaMemcpy(h_output, d_output, sizeof(float)*block_num, cudaMemcpyDeviceToHost);

    if (check(h_output, res, block_num)) {
        std::cout << "Success!" << std::endl;
    } else {
        std::cout << "Failed!" << std::endl;
    }
    free(h_input);
    free(h_output);
    free(res);
    cudaFree(d_input);
    cudaFree(d_output);

    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}