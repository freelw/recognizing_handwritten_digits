#include "cuda_ops.h"

#ifndef GCC_ASAN

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

void* CUDAOps::alloc(size_t size) {
    assert(false); // Not implemented yet
}

void CUDAOps::memset(void* ptr, int value, size_t size) {
    assert(false); // Not implemented yet
}

void CUDAOps::memcpy(void* dst, const void* src, size_t size) {
    assert(false); // Not implemented yet
}

void CUDAOps::free(void* ptr) {
    assert(false); // Not implemented yet
}

void CUDAOps::cp_to_device(Tensor *dst_tensor, char *src, size_t size) {
    assert(false); // Not implemented yet
}

void CUDAOps::cp_from_device(char *dst, Tensor *src_tensor, size_t size) {
    assert(false); // Not implemented yet
}
#endif // GCC_ASAN