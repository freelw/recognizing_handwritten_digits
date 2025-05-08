#include "cuda_ops.h"

#ifndef GCC_ASAN

#include <cuda.h>
#include <cuda_runtime.h>

void CUDAOps::add(Tensor *lhs, const Tensor *rhs, Tensor *res) {
    assert(false); // Not implemented yet
}

void CUDAOps::addEq(Tensor *lhs, const Tensor *rhs) {
    assert(false); // Not implemented yet
}

void CUDAOps::expandAdd(Tensor *lhs, const Tensor *rhs, Tensor *res) {
    assert(false); // Not implemented yet
}

void CUDAOps::at(Tensor *lhs, const Tensor *rhs, Tensor *res) {
    assert(false); // Not implemented yet
}

void CUDAOps::mul(Tensor *lhs, const Tensor *rhs, Tensor *res) {
    assert(false); // Not implemented yet
}

void CUDAOps::sum(Tensor *lhs, Tensor *res, int dim) {
    assert(false); // Not implemented yet
}

void CUDAOps::relu(Tensor *lhs, Tensor *res) {
    assert(false); // Not implemented yet
}

void CUDAOps::reluPrime(Tensor *lhs, Tensor *res) {
    assert(false); // Not implemented yet
}

void CUDAOps::crossEntropy(Tensor *lhs, const Tensor *labels, Tensor *maxs, Tensor *sums, Tensor *res) {
    assert(false); // Not implemented yet
}

void CUDAOps::crossEntropyBackward(Tensor *lhs, const Tensor *labels, Tensor *maxs, Tensor *sums, Tensor *res) {
    assert(false); // Not implemented yet
}

void CUDAOps::calcAllGradNorm(const std::vector<Tensor*> &grads, Tensor *norm) {
    assert(false); // Not implemented yet
}

void CUDAOps::clipGrad(Tensor *grad, const Tensor *norm, float grad_clip_val) {
    assert(false); // Not implemented yet
}

void CUDAOps::adamStep(Tensor *w, Tensor *grad, Tensor *m, Tensor *v, int t, float lr, float beta1, float beta2, float epsilon) {
    assert(false); // Not implemented yet
}

void CUDAOps::init_weight_gauss(Tensor *tensor, float mean, float sigma) {
    assert(false); // Not implemented yet
}

void CUDAOps::init_weight_uniform(Tensor *tensor, float sigma) {
    assert(false); // Not implemented yet
}

void CUDAOps::fill(Tensor *tensor, float value) {
    
}

void* CUDAOps::alloc(size_t size) {
    void *ret = nullptr;
    cudaMalloc((void **)&ret, size);
    return ret;
}

void CUDAOps::memset(void* ptr, int value, size_t size) {
    ::cudaMemset(ptr, value, size);
}

void CUDAOps::cp_device_to_device(void* dst, const void* src, size_t size) {
    ::cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}

void CUDAOps::free(void* ptr) {
    ::cudaFree(ptr);
}

void CUDAOps::cp_to_device(Tensor *dst_tensor, char *src, size_t size) {
    ::cudaMemcpy(dst_tensor->get_data(), src, size, cudaMemcpyHostToDevice);
}

void CUDAOps::cp_from_device(char *dst, Tensor *src_tensor, size_t size) {
    ::cudaMemcpy(dst, src_tensor->get_data(), size, cudaMemcpyDeviceToHost);
}
#endif // GCC_ASAN