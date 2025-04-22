#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

__global__ void reduce(float *d_input, float *d_output) {

}

int main() {
    
    dim3 gridDim(1);
    dim3 blockDim(1);

    reduce<<<gridDim, blockDim>>>(NULL, NULL);
    return 0;
}