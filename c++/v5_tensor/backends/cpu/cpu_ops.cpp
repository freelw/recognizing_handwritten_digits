#include "cpu_ops.h"
#include <string.h>
#include <cmath>

void CPUOps::add(Tensor *lhs, const Tensor *rhs, Tensor *res) {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    
    int size = lhs->size();
    for (int i = 0; i < size; ++i) {
        static_cast<float*>(res->get_data())[i] = 
            static_cast<float*>(lhs->get_data())[i] + 
            static_cast<float*>(rhs->get_data())[i];
    }
}

void CPUOps::addEq(Tensor *lhs, const Tensor *rhs) {
    
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    
    int size = lhs->size();
    for (int i = 0; i < size; ++i) {
        static_cast<float*>(lhs->get_data())[i] += 
            static_cast<float*>(rhs->get_data())[i];
    }
}

void CPUOps::expandAdd(Tensor *lhs, const Tensor *rhs, Tensor *res) {

    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);

    int size = lhs->size();
    auto shape = lhs->get_shape();
    assert(shape.size() == 2);
    assert(rhs->get_shape().size() == 1);   
    assert(rhs->get_shape()[0] == shape[1]);

    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            static_cast<float*>(res->get_data())[i * shape[1] + j] = 
                static_cast<float*>(lhs->get_data())[i * shape[1] + j] + 
                static_cast<float*>(rhs->get_data())[j];
        }
    }
}

void CPUOps::at(Tensor *lhs, const Tensor *rhs, Tensor *res) {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);

    auto lshape = lhs->get_shape();
    auto rshape = rhs->get_shape();
    auto res_shape = res->get_shape();
    assert(lshape.size() == 2);
    assert(rshape.size() == 2);
    assert(res_shape.size() == 2);
    assert(lshape[1] == rshape[0]);
    assert(res_shape[0] == lshape[0]);
    assert(res_shape[1] == rshape[1]);

    for (int i = 0; i < lshape[0]; ++i) {
        for (int j = 0; j < rshape[1]; ++j) {
            static_cast<float*>(res->get_data())[i * rshape[1] + j] = 0;
            for (int k = 0; k < lshape[1]; ++k) {
                static_cast<float*>(res->get_data())[i * rshape[1] + j] += 
                    static_cast<float*>(lhs->get_data())[i * lshape[1] + k] * 
                    static_cast<float*>(rhs->get_data())[k * rshape[1] + j];
            }
        }
    }
}

void CPUOps::mul(Tensor *lhs, const Tensor *rhs, Tensor *res) {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);

    int size = lhs->size();
    for (int i = 0; i < size; ++i) {
        static_cast<float*>(res->get_data())[i] = 
            static_cast<float*>(lhs->get_data())[i] * 
            static_cast<float*>(rhs->get_data())[i];
    }
}

// void sum(Tensor *lhs, Tensor *res, int dim) override;
void CPUOps::sum(Tensor *lhs, Tensor *res, int dim) {
    assert(lhs != nullptr);
    assert(res != nullptr);
    assert(dim >= 0 && dim < lhs->get_rank());

    auto shape = lhs->get_shape();
    auto res_shape = res->get_shape();
    assert(dim == 0);

    assert(shape.size() == res_shape.size());
    assert(shape[1] == res_shape[1]);
    assert(res_shape[0] == 1);

    for (int i = 0; i < shape[1]; ++i) {
        static_cast<float*>(res->get_data())[i] = 0;
        for (int j = 0; j < shape[0]; ++j) {
            static_cast<float*>(res->get_data())[i] += 
                static_cast<float*>(lhs->get_data())[j * shape[1] + i];
        }
    }
}

// void relu(Tensor *lhs, Tensor *res) override;
void CPUOps::relu(Tensor *lhs, Tensor *res) {
    assert(lhs != nullptr);
    assert(res != nullptr);

    int size = lhs->size();
    for (int i = 0; i < size; ++i) {
        static_cast<float*>(res->get_data())[i] = 
            std::max(0.0f, static_cast<float*>(lhs->get_data())[i]);
    }
}

// void reluPrime(Tensor *lhs, Tensor *res) override;
void CPUOps::reluPrime(Tensor *lhs, Tensor *res) {
    assert(lhs != nullptr);
    assert(res != nullptr);

    int size = lhs->size();
    for (int i = 0; i < size; ++i) {
        static_cast<float*>(res->get_data())[i] = 
            static_cast<float*>(lhs->get_data())[i] > 0 ? 1.0f : 0.0f;
    }
}

void CPUOps::crossEntropy(Tensor *lhs, const Tensor *labels, Tensor *maxs, Tensor *sums, Tensor *res) {
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

    int batch_size = lhs->get_shape()[0];
    int size = lhs->get_shape()[1];
    float loss_value = 0;

    // maxs and sums are out params
    float *data = static_cast<float*>(lhs->get_data());
    for (int j = 0; j < batch_size; ++j) {
        float max = data[j * size];
        for (int i = 0; i < size; ++i) {
            auto e = data[j * size + i];
            if (max < e) {
                max = e;
            }
        }
        float sum = 0;
        auto target = static_cast<int32_t*>(labels->get_data())[j];
        float zt = data[j * size + target];
        for (int i = 0; i < size; ++i) {
            float e = data[j* size + i];
            e = std::exp(e - max);
            sum += e;
        }
        static_cast<float*>(maxs->get_data())[j] = max;
        static_cast<float*>(sums->get_data())[j] = sum;
        loss_value += -(zt - max - std::log(sum));
    }
    static_cast<float*>(res->get_data())[0] = loss_value / batch_size;
}

void CPUOps::crossEntropyBackward(Tensor *lhs, const Tensor *labels, Tensor *maxs, Tensor *sums, Tensor *res) {
    assert(lhs != nullptr);
    assert(labels != nullptr);
    assert(maxs != nullptr);
    assert(sums != nullptr);
    assert(res != nullptr);

    
    int batch_size = lhs->get_shape()[0];
    int size = lhs->get_shape()[1];
    float *data = static_cast<float*>(lhs->get_data());
    float *res_data = static_cast<float*>(res->get_data());

    for (int j = 0; j < batch_size; ++j) {
        float max = static_cast<float*>(maxs->get_data())[j];
        float sum = static_cast<float*>(sums->get_data())[j];
        auto target = static_cast<int32_t*>(labels->get_data())[j];
        for (int i = 0; i < size; ++i) {
            if (i == target) {
                res_data[j * size + i] = 
                    (std::exp(data[j * size + i] - max) / sum - 1) / batch_size;
            } else {
                res_data[j * size + i] = 
                    (std::exp(data[j * size + i] - max) / sum) / batch_size;
            }
        }
    }
}

void* CPUOps::alloc(size_t size) {
    return malloc(size);
}

void CPUOps::memset(void* ptr, int value, size_t size) {
    ::memset(ptr, value, size);
}

void CPUOps::memcpy(void* dst, const void* src, size_t size) {
    ::memcpy(dst, src, size);
}

void CPUOps::free(void* ptr) {
    ::free(ptr);
}