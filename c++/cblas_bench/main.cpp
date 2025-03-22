#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <string.h>

void init_matrix(float *A, int size, float sigma) {
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w(seed1);
    std::normal_distribution<float> distribution_w(0.0, sigma);
    for (int i = 0; i < size; ++i) {
        A[i] = distribution_w(generator_w);
    }
}

float *alloc_matrix(int size) {
    return new float[size];
}

void free_matrix(float *A) {
    delete []A;
}

int main(void) {
    int lda = 256;

    float *A = alloc_matrix(256*256);

    int ldb = 256;

    float *B = alloc_matrix(256*256);

    int ldc = 256;

    float *C = alloc_matrix(256*256);

    init_matrix(A, 256*256, 0.1);
    init_matrix(B, 256*256, 0.1);

    for (int i = 0; i < 10000; ++ i) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256, 256, 256, 1.0, A, lda,
                B, ldb, 0.0, C, ldc);
        memcpy(A, C, 256*256*sizeof(float));
    }

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

  return 0;
}