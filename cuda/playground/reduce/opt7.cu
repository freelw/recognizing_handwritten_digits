#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

#define WARP_SIZE 32
#define THREAD_PER_BLOCK 256

template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (blockSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}

template <unsigned int blockSize, int NUM_PER_THREAD>
__global__ void reduce(float *d_in,float *d_out, unsigned int n){
    float sum = 0;

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * NUM_PER_THREAD) + threadIdx.x;

    #pragma unroll
    for(int iter=0; iter<NUM_PER_THREAD; iter++){
        sum += d_in[i+iter*blockSize];
    }
    
    // Shared mem for partial sums (one per warp in the block)
    static __shared__ float warpLevelSums[WARP_SIZE]; 
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;

    sum = warpReduceSum<blockSize>(sum);

    if(laneId == 0 )warpLevelSums[warpId] = sum;
    __syncthreads();
    // read from shared memory only if that warp existed
    sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;
    // Final reduce using first warp
    if (warpId == 0) sum = warpReduceSum<blockSize/WARP_SIZE>(sum); 
    // write result for this block to global mem
    if (tid == 0) d_out[blockIdx.x] = sum;
}

bool check(float *h_output, float *res, int size) {
    for (int i = 0; i < size; ++i) {
        if (fabs(h_output[i] - res[i]) > 1e-2) {
            std::cout << "Error: " << "[" << i << "] " << h_output[i] << " != " << res[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 32 * 1024 * 1024;
    const int size = N * sizeof(float);
    
    const int block_num = 1024;
    const int NUM_PER_BLOCK = N / block_num;
    const int NUM_PER_THREAD = NUM_PER_BLOCK/THREAD_PER_BLOCK;
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
        for (int j = 0; j < NUM_PER_BLOCK; ++ j) {
            res[i] += h_input[i * NUM_PER_BLOCK + j];
        }
    }
    
    dim3 gridDim(block_num);
    dim3 blockDim(THREAD_PER_BLOCK);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    reduce<THREAD_PER_BLOCK, NUM_PER_THREAD><<<gridDim, blockDim>>>(d_input, d_output, N);
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
    return 0;
}