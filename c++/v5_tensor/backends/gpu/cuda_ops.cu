#include "cuda_ops.h"

#ifndef GCC_ASAN

#include "kernel.cuh"
#include <random>
#include <chrono>

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

    assert(lhs->get_dim() == 2);

    dim3 gridDim(
        (lshape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        (lshape[0] + TILE_WIDTH - 1) / TILE_WIDTH
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
    
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    
    auto lshape = lhs->get_shape();
    auto rshape = rhs->get_shape();
    
    assert(lshape == rshape);

    auto lstrides = lhs->get_strides();
    auto rstrides = rhs->get_strides();

    int rank = lhs->get_dim();

    assert(rank <= 2);

    if (rank == 1) {
        dim3 gridDim(
            (lshape[0] + TILE_WIDTH - 1) / TILE_WIDTH
        );
        dim3 blockDim(TILE_WIDTH);
        tensor_add_eq_1d<<<gridDim, blockDim>>>(
            (float *)lhs->get_data(),
            (float *)rhs->get_data(),
            lshape[0]
        );
    } else if (rank == 2) {
        dim3 gridDim(
            (lshape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
            (lshape[0] + TILE_WIDTH - 1) / TILE_WIDTH
        );
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        tensor_add_eq_2d<<<gridDim, blockDim>>>(
            (float *)lhs->get_data(),
            (float *)rhs->get_data(),
            lshape[0],
            lshape[1],
            lstrides[0],
            lstrides[1],
            rstrides[0],
            rstrides[1]
        );
    }
}

void CUDAOps::expandAdd(Tensor *lhs, const Tensor *rhs, Tensor *res) {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);

    int size = lhs->size();
    auto shape = lhs->get_shape();
    assert(shape.size() == 2);
    assert(rhs->get_shape().size() == 1);   
    assert(rhs->get_shape()[0] == shape[1]);
    assert(shape == res->get_shape());

    auto lstrides = lhs->get_strides();
    auto res_strides = res->get_strides();

    dim3 gridDim(
        (shape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        (shape[0] + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

    expand_add<<<gridDim, blockDim>>>(
        (float *)lhs->get_data(),
        (float *)rhs->get_data(),
        (float *)res->get_data(),
        shape[0],
        shape[1],
        lstrides[0],
        lstrides[1],
        res_strides[0],
        res_strides[1]
    );
}

void CUDAOps::at(Tensor *lhs, const Tensor *rhs, Tensor *res) {
    auto lshape = lhs->get_shape();
    auto rshape = rhs->get_shape();
    auto res_shape = res->get_shape();

    auto lstrides = lhs->get_strides();
    auto rstrides = rhs->get_strides();
    auto res_strides = res->get_strides();

    assert(lhs->get_dim() == 2);
    assert(rhs->get_dim() == 2);
    assert(res->get_dim() == 2);

    assert(lshape[1] == rshape[0]);
    assert(res_shape[0] == lshape[0]);
    assert(res_shape[1] == rshape[1]);

    const int M = lshape[0];
    const int N = lshape[1];
    const int P = rshape[1];

    const int stride_M0 = lstrides[0];
    const int stride_M1 = lstrides[1];
    const int stride_N0 = rstrides[0];
    const int stride_N1 = rstrides[1];
    const int stride_P0 = res_strides[0];
    const int stride_P1 = res_strides[1];

    this->memset((float *)res->get_data(), 0, res->size());

    dim3 gridDim(
        (P + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH
    );
    dim3 blockDim(
        TILE_WIDTH,
        TILE_WIDTH
    );

    tensor_at_2d<<<gridDim, blockDim>>>(
        (float *)lhs->get_data(),
        (float *)rhs->get_data(),
        (float *)res->get_data(),
        M,
        N,
        P,
        stride_M0,
        stride_M1,
        stride_N0,
        stride_N1,
        stride_P0,
        stride_P1
    );
}

void CUDAOps::emb_at(Tensor *lhs, const Tensor *indices, const Tensor *rhs, Tensor *res) {
    assert(false);
}

void CUDAOps::mul(Tensor *lhs, const Tensor *rhs, Tensor *res) {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    assert(lhs->get_dim() == 2);

    auto lshape = lhs->get_shape();
    auto rshape = rhs->get_shape();
    auto res_shape = res->get_shape();

    assert(lshape == rshape);
    assert(res_shape == lshape);

    auto lstrides = lhs->get_strides();
    auto rstrides = rhs->get_strides();
    auto res_strides = res->get_strides();

    dim3 gridDim(
        (lshape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        (lshape[0] + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

    tensor_mul_2d<<<gridDim, blockDim>>>(
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

void CUDAOps::sum(Tensor *lhs, Tensor *res, int dim) {
    assert(lhs != nullptr);
    assert(res != nullptr);
    assert(dim >= 0 && dim < lhs->get_dim());

    auto shape = lhs->get_shape();
    auto res_shape = res->get_shape();
    assert(dim == 0);
    auto lstrides = lhs->get_strides();
    assert(lhs->get_dim() == 2);
    assert(res->get_dim() == 1);

    dim3 gridDim(
        (shape[1] + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH);

    tensor_sum_2d_dim0<<<gridDim, blockDim>>>(
        (float *)lhs->get_data(),
        (float *)res->get_data(),
        shape[0],
        shape[1],
        lstrides[0],
        lstrides[1]
    );
}

void CUDAOps::relu(Tensor *lhs, Tensor *res) {
    assert(lhs != nullptr);
    assert(res != nullptr);

    int length = lhs->length();

    auto shape = lhs->get_shape();
    auto res_shape = res->get_shape();
    assert(shape == res_shape);

    dim3 gridDim(
        (length + TILE_WIDTH - 1) / TILE_WIDTH
    );
    dim3 blockDim(TILE_WIDTH);

    tensor_relu<<<gridDim, blockDim>>>(
        (float *)lhs->get_data(),
        (float *)res->get_data(),
        length
    );
}

void CUDAOps::reluPrime(Tensor *lhs, Tensor *res) {
    assert(lhs != nullptr);
    assert(res != nullptr);

    int length = lhs->length();

    auto shape = lhs->get_shape();
    auto res_shape = res->get_shape();
    assert(shape == res_shape);

    dim3 gridDim(
        (length + TILE_WIDTH - 1) / TILE_WIDTH
    );
    dim3 blockDim(TILE_WIDTH);

    tensor_relu_prime<<<gridDim, blockDim>>>(
        (float *)lhs->get_data(),
        (float *)res->get_data(),
        length
    );
}

void CUDAOps::crossEntropy(Tensor *lhs, const Tensor *labels, Tensor *maxs, Tensor *sums, Tensor *res) {
    assert(lhs != nullptr);
    assert(labels != nullptr);
    assert(maxs != nullptr);
    assert(sums != nullptr);
    assert(res != nullptr);

    assert(lhs->get_shape().size() == 2);
    assert(labels->get_shape().size() == 1);
    assert(maxs->get_shape().size() == 1);
    assert(sums->get_shape().size() == 1);
    assert(res->get_shape().size() == 1);
    assert(lhs->get_shape()[0] == labels->get_shape()[0]);
    assert(lhs->get_shape()[0] == maxs->get_shape()[0]);
    assert(lhs->get_shape()[0] == sums->get_shape()[0]);
    assert(res->get_shape()[0] == 1);

    auto lstrides = lhs->get_strides();

    this->memset((float *)res->get_data(), 0, res->size());
    this->memset((float *)maxs->get_data(), 0, maxs->size());
    this->memset((float *)sums->get_data(), 0, sums->size());

    dim3 gridDim(
        (lhs->get_shape()[0] + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH);

    // std::cout << "lhs->get_shape()[0]" << lhs->get_shape()[0] << std::endl;
    // std::cout << "lhs->get_shape()[1]" << lhs->get_shape()[1] << std::endl;
    // std::cout << "lstrides[0]" << lstrides[0] << std::endl;
    // std::cout << "lstrides[1]" << lstrides[1] << std::endl;

    cross_entropy<<<gridDim, blockDim, TILE_WIDTH*sizeof(float)>>>(
        (float *)lhs->get_data(),
        (int32_t *)labels->get_data(),
        (float *)maxs->get_data(),
        (float *)sums->get_data(),
        (float *)res->get_data(),
        lhs->get_shape()[0],
        lhs->get_shape()[1],
        lstrides[0],
        lstrides[1]
    );
}

void CUDAOps::crossEntropyBackward(Tensor *lhs, const Tensor *labels, Tensor *maxs, Tensor *sums, Tensor *res) {
    assert(lhs != nullptr);
    assert(labels != nullptr);
    assert(maxs != nullptr);
    assert(sums != nullptr);
    assert(res != nullptr);

    int batch_size = lhs->get_shape()[0];
    int size = lhs->get_shape()[1];
    float *data = static_cast<float*>(lhs->get_data());
    float *res_data = static_cast<float*>(res->get_data());
    auto lstrides = lhs->get_strides();
    auto res_strides = res->get_strides();
    assert(lstrides.size() == 2);
    assert(res_strides.size() == 2);

    dim3 gridDim(
        (batch_size + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH);

    cross_entropy_backward<<<gridDim, blockDim>>>(
        (float *)lhs->get_data(),
        (int32_t *)labels->get_data(),
        (float *)maxs->get_data(),
        (float *)sums->get_data(),
        (float *)res->get_data(),
        batch_size,
        size,
        lstrides[0],
        lstrides[1],
        res_strides[0],
        res_strides[1]
    );
}

void CUDAOps::calcAllGradNorm(const std::vector<Tensor*> &grads, Tensor *norm) {
    assert(norm != nullptr);
    assert(norm->length() == 1);
    this->memset((float *)norm->get_data(), 0, norm->size());
    for (auto &grad : grads) {
        assert(grad != nullptr);
        auto length = grad->length();
        dim3 gridDim(
            (length + TILE_WIDTH - 1) / TILE_WIDTH
        );
        dim3 blockDim(TILE_WIDTH);
        tensor_l2_norm<<<gridDim, blockDim, TILE_WIDTH*sizeof(float)>>>(
            (float *)grad->get_data(),
            (float *)norm->get_data(),
            length
        );
    }
}

void CUDAOps::clipGrad(Tensor *grad, const Tensor *norm, float grad_clip_val) {
    assert(grad != nullptr);
    assert(norm != nullptr);

    assert(norm->get_shape().size() == 1);

    auto length = grad->length();
    auto norm_length = norm->length();
    assert(norm_length == 1);

    dim3 gridDim(
        (length + TILE_WIDTH - 1) / TILE_WIDTH
    );
    dim3 blockDim(TILE_WIDTH);

    tensor_clip<<<gridDim, blockDim>>>(
        (float *)grad->get_data(),
        (float *)norm->get_data(),
        length,
        grad_clip_val
    );
}

void CUDAOps::adamStep(Tensor *w, Tensor *grad, Tensor *m, Tensor *v, int t, float lr, float beta1, float beta2, float epsilon) {
    assert(w != nullptr);
    assert(grad != nullptr);
    assert(m != nullptr);
    assert(v != nullptr);

    assert(!w->is_view());
    assert(!grad->is_view());
    assert(!m->is_view());
    assert(!v->is_view());

    assert(w->get_shape() == grad->get_shape());
    assert(w->get_shape() == m->get_shape());
    assert(w->get_shape() == v->get_shape());

    auto length = w->length();

    dim3 gridDim(
        (length + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH);

    tensor_adam_step<<<gridDim, blockDim>>>(
        (float *)w->get_data(),
        (float *)grad->get_data(),
        (float *)m->get_data(),
        (float *)v->get_data(),
        length,
        t,
        beta1,
        beta2,
        lr,
        epsilon
    );
}

void CUDAOps::init_weight_gauss(Tensor *tensor, float mean, float sigma) {
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w(seed1);
    std::normal_distribution<float> distribution_w(0.0, sigma);
    auto size = tensor->size();
    float *data = static_cast<float*>(::malloc(size));
    for (int i = 0; i < tensor->length(); ++i) {
        data[i] = distribution_w(generator_w) + mean;
    }
    this->cp_to_device(tensor, (char *)data, size);
    ::free(data);
}

void CUDAOps::init_weight_uniform(Tensor *tensor, float sigma) {
    assert(false); // Not implemented yet
}

void CUDAOps::init_weight_for_dbg(Tensor *tensor) {
    auto size = tensor->size();
    void *_data = ::malloc(size);

    if (tensor->get_dtype() == FLOAT32) {
        float *data = static_cast<float*>(_data);
        for (int i = 0; i < tensor->length(); ++i) {
            data[i] = static_cast<float>(i) * 1e-5;
        }
    } else if (tensor->get_dtype() == INT32) {
        int32_t *data = static_cast<int32_t*>(_data);
        for (int i = 0; i < tensor->length(); ++i) {
            data[i] = i % 10;
        }
    } else {
        assert(false);
    }
    this->cp_to_device(tensor, (char *)_data, size);
    ::free(_data);
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

void CUDAOps::reshape_deep_cp(Tensor *dst_tensor, const Tensor *src_tensor, const Tensor *src_strides, int _dim) {
    assert(false); // Not implemented yet
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

void CUDAOps::cp_from_device(char *dst, const Tensor *src_tensor, size_t size) {
    ::cudaMemcpy(dst, src_tensor->get_data(), size, cudaMemcpyDeviceToHost);
}
#endif // GCC_ASAN