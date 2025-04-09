#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

#define THREAD_PER_BLOCK 256

__global__ void reduce(float *d_input, float *d_output) {

}

int main() {
    const int N = 32 * 1024 * 1024;
    const int size = N * sizeof(float);
    float *h_input = (float *)malloc(size);
    float *d_input;
    cudaMalloc((void **)&d_input, size);
    int block_num = N / THREAD_PER_BLOCK;
    float *h_output = (float *)malloc(sizeof(float)*block_num);
    float *res = (float *)malloc(sizeof(float)*block_num);
    memset(res, 0, sizeof(float)*block_num);
    float *d_output;
    cudaMalloc((void **)&d_output, sizeof(float)*block_num);
    for (int i = 0; i < N; i++) {
        h_input[i] = 2 * (float)drand48() - 1.0;
    }

    for (int i = 0; i < block_num; ++ i) {
        for (int j = 0; j < THREAD_PER_BLOCK; ++ j) {
            res[i] += h_input[i * THREAD_PER_BLOCK + j];
        }
    }
    dim3 grid(block_num);
    dim3 block(THREAD_PER_BLOCK);

    reduce<<<grid, block>>>(d_input, d_output);
    cudaMemcpy(h_input, d_input, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output, d_output, sizeof(float)*block_num, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}