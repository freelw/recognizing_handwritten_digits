#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <curand.h>

// CUDA API error checking
#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);       \
      throw std::runtime_error("CUDA error");                                  \
    }                                                                          \
  } while (0)

// curand API error checking
#define CURAND_CHECK(err)                                                      \
  do {                                                                         \
    curandStatus_t err_ = (err);                                               \
    if (err_ != CURAND_STATUS_SUCCESS) {                                       \
      std::printf("curand error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
      throw std::runtime_error("curand error");                                \
    }                                                                          \
  } while (0)

int main() {
    float *d_data = nullptr;
    float *h_data = nullptr;
    const int L =  12;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_data), L * sizeof(float)));
    h_data = static_cast<float *>(::malloc(L * sizeof(float)));
    curandGenerator_t gen;
    curandOrdering_t order = CURAND_ORDERING_PSEUDO_DEFAULT;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937));
    CURAND_CHECK(curandSetGeneratorOrdering(gen, order));
    const unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
    while(1) {
        CURAND_CHECK(curandGenerateUniform(gen, d_data, L));
        cudaMemcpy(h_data, d_data, L * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < L; ++i) {
            std::printf("%f ", h_data[i]);
        }
    }
    CURAND_CHECK(curandDestroyGenerator(gen));
    ::free(h_data);
    CUDA_CHECK(cudaFree(d_data));
    return 0;
}