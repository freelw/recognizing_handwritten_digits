#include "cuda_ops.h"

#ifndef GCC_ASAN

#include "kernel.cuh"

void CUDAOps::add(Tensor *lhs, const Tensor *rhs, Tensor *res) {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);

    auto lshape = lhs->get_shape();
    auto rshape = rhs->get_shape();
    auto res_shape = res->get_shape();

    assert(lshape == rshape);
    assert(res_shape == lshape);

    auto lstrides = lhs->get_strides();
    auto rstrides = rhs->get_strides();
    auto res_strides = res->get_strides();

    assert(lhs->get_rank() == 2);

    dim3 gridDim(
        (lshape[0] + TILE_WIDTH - 1) / TILE_WIDTH,
        (lshape[1] + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

    tensor_add_2d<<<gridDim, blockDim>>>(
        (float *)lhs->get_data(),
        (float *)rhs->get_data(),
        (float *)res->get_data(),
        lshape[0],
        lshape[1],
        lstrides[0],
        lstrides[1],
        rstrides[0],
        rstrides[1],
        res_strides[0],
        res_strides[1]
    );
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
    dim3 gridDim(
        (tensor->length() + TILE_WIDTH - 1) / TILE_WIDTH
    );
    dim3 blockDim(TILE_WIDTH);
    fill_float<<<gridDim, blockDim>>>(
        (float *)tensor->get_data(),
        tensor->length(),
        value
    );
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