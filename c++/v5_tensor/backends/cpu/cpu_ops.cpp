#include "cpu_ops.h"
#include <string.h>
#include <cmath>
#include <random>
#include <chrono>

CPUOps::CPUOps() {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    gen = std::mt19937(seed);
    dis = std::uniform_real_distribution<>(0, 1);
}

void CPUOps::add(Tensor *lhs, const Tensor *rhs, Tensor *res) {
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
    
    for (int i = 0; i < lshape[0]; ++i) {
        for (int j = 0; j < lshape[1]; ++j) {
            static_cast<float*>(res->get_data())[i * res_strides[0] + j * res_strides[1]] = 
                static_cast<float*>(lhs->get_data())[i * lstrides[0] + j * lstrides[1]] + 
                static_cast<float*>(rhs->get_data())[i * rstrides[0] + j * rstrides[1]];
        }
    }
}

void CPUOps::addEq(Tensor *lhs, const Tensor *rhs) {
    
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    
    auto lshape = lhs->get_shape();
    auto rshape = rhs->get_shape();
    
    assert(lshape == rshape);

    auto lstrides = lhs->get_strides();
    auto rstrides = rhs->get_strides();

    int dim = lhs->get_dim();

    assert(dim <= 3);
    if (dim == 1) {
        for (int i = 0; i < lshape[0]; ++i) {
            static_cast<float*>(lhs->get_data())[i * lstrides[0]] += 
                static_cast<float*>(rhs->get_data())[i * rstrides[0]];
        }
    } else if (dim == 2) {
        for (int i = 0; i < lshape[0]; ++i) {
            for (int j = 0; j < lshape[1]; ++j) {
                static_cast<float*>(lhs->get_data())[i * lstrides[0] + j * lstrides[1]] += 
                    static_cast<float*>(rhs->get_data())[i * rstrides[0] + j * rstrides[1]];
            }
        }
    } else if (dim == 3) {
        for (int i = 0; i < lshape[0]; ++i) {
            for (int j = 0; j < lshape[1]; ++j) {
                for (int k = 0; k < lshape[2]; ++k) {
                    static_cast<float*>(lhs->get_data())[i * lstrides[0] + j * lstrides[1] + k * lstrides[2]] += 
                        static_cast<float*>(rhs->get_data())[i * rstrides[0] + j * rstrides[1] + k * rstrides[2]];
                }
            }
        }
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
    assert(shape == res->get_shape());

    auto lstrides = lhs->get_strides();
    auto res_strides = res->get_strides();

    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            static_cast<float*>(res->get_data())[i * res_strides[0] + j * res_strides[1]] = 
                static_cast<float*>(lhs->get_data())[i * lstrides[0] + j * lstrides[1]] + 
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

    float *res_data = static_cast<float*>(res->get_data());
    float *lhs_data = static_cast<float*>(lhs->get_data());
    float *rhs_data = static_cast<float*>(rhs->get_data());

    auto lstrides = lhs->get_strides();
    auto rstrides = rhs->get_strides();
    auto res_strides = res->get_strides();

    for (int i = 0; i < lshape[0]; ++i) {
        for (int j = 0; j < rshape[1]; ++j) {
            res_data[i * res_strides[0] + j * res_strides[1]] = 0;
            for (int k = 0; k < lshape[1]; ++k) {
                res_data[i * res_strides[0] + j * res_strides[1]] += 
                    lhs_data[i * lstrides[0] + k * lstrides[1]] * 
                    rhs_data[k * rstrides[0] + j * rstrides[1]];
            }
        }
    }
}

void CPUOps::emb_at(Tensor *lhs, const Tensor *indices, const Tensor *rhs, Tensor *res) {
    assert(false);
}

void CPUOps::mul(Tensor *lhs, const Tensor *rhs, Tensor *res) {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    assert(lhs->get_dim() == 2 || lhs->get_dim() == 1);

    auto lshape = lhs->get_shape();
    auto rshape = rhs->get_shape();
    auto res_shape = res->get_shape();

    assert(lshape == rshape);
    assert(res_shape == lshape);

    auto lstrides = lhs->get_strides();
    auto rstrides = rhs->get_strides();
    auto res_strides = res->get_strides();

    if (lhs->get_dim() == 1) {
        for (int i = 0; i < lshape[0]; ++i) {
            static_cast<float*>(res->get_data())[i] = 
                static_cast<float*>(lhs->get_data())[i * lstrides[0]] * 
                static_cast<float*>(rhs->get_data())[i * rstrides[0]];
        }
    } else if (lhs->get_dim() == 2) {
        for (int i = 0; i < lshape[0]; ++i) {
            for (int j = 0; j < lshape[1]; ++j) {
                static_cast<float*>(res->get_data())[i * res_strides[0] + j * res_strides[1]] = 
                    static_cast<float*>(lhs->get_data())[i * lstrides[0] + j * lstrides[1]] * 
                    static_cast<float*>(rhs->get_data())[i * rstrides[0] + j * rstrides[1]];
            }
        }
    }
}

void CPUOps::sum(Tensor *lhs, Tensor *res, int dim) {
    assert(lhs != nullptr);
    assert(res != nullptr);
    assert(dim >= 0 && dim < lhs->get_dim());

    auto shape = lhs->get_shape();
    auto res_shape = res->get_shape();
    assert(dim == 0);

    auto lstrides = lhs->get_strides();
    assert(lhs->get_dim() == 2);
    assert(res->get_dim() == 1);

    for (int i = 0; i < shape[1]; ++i) {
        static_cast<float*>(res->get_data())[i] = 0;
        for (int j = 0; j < shape[0]; ++j) {
            static_cast<float*>(res->get_data())[i] += 
                static_cast<float*>(lhs->get_data())[j * lstrides[0] + i * lstrides[1]];
        }
    }
}

void CPUOps::relu(Tensor *lhs, Tensor *res) {
    assert(lhs != nullptr);
    assert(res != nullptr);

    int length = lhs->length();
    for (int i = 0; i < length; ++i) {
        static_cast<float*>(res->get_data())[i] = 
            std::max(0.0f, static_cast<float*>(lhs->get_data())[i]);
    }
}

void CPUOps::reluPrime(Tensor *lhs, Tensor *res) {
    assert(lhs != nullptr);
    assert(res != nullptr);

    int length = lhs->length();
    for (int i = 0; i < length; ++i) {
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
    int32_t *labels_data = static_cast<int32_t*>(labels->get_data());
    auto lstrides = lhs->get_strides();
    for (int j = 0; j < batch_size; ++j) {
        float max = data[j * lstrides[0]];
        for (int i = 0; i < size; ++i) {
            auto e = data[j * lstrides[0] + i * lstrides[1]];
            if (max < e) {
                max = e;
            }
        }
        float sum = 0;
        auto target = labels_data[j];
        assert(target >= 0 && target < size);
        float zt = data[j * lstrides[0] + target * lstrides[1]];
        for (int i = 0; i < size; ++i) {
            float e = data[j* lstrides[0] + i * lstrides[1]];
            e = std::exp(e - max);
            sum += e;
        }
        static_cast<float*>(maxs->get_data())[j] = max;
        static_cast<float*>(sums->get_data())[j] = sum;
        loss_value += -(zt - max - std::log(sum));
    }
    static_cast<float*>(res->get_data())[0] = loss_value;
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
    auto lstrides = lhs->get_strides();
    auto res_strides = res->get_strides();
    assert(lstrides.size() == 2);
    assert(res_strides.size() == 2);

    for (int j = 0; j < batch_size; ++j) {
        float max = static_cast<float*>(maxs->get_data())[j];
        float sum = static_cast<float*>(sums->get_data())[j];
        auto target = static_cast<int32_t*>(labels->get_data())[j];
        for (int i = 0; i < size; ++i) {
            if (i == target) {
                res_data[j * res_strides[0] + i * res_strides[1]] = 
                    (std::exp(data[j * lstrides[0] + i * lstrides[1]] - max) / sum - 1) / batch_size;
            } else {
                res_data[j * res_strides[0] + i * res_strides[1]] = 
                    (std::exp(data[j * lstrides[0] + i * lstrides[1]] - max) / sum) / batch_size;
            }
        }
    }
}

void CPUOps::div(Tensor *dst, Tensor *src, float value) {
    assert(dst->length() == src->length());
    auto length = dst->length();
    for (int i = 0; i < length; ++i) {
        static_cast<float*>(dst->get_data())[i] = 
            static_cast<float*>(src->get_data())[i] / value;
    }
}

void CPUOps::build_dropout_mask(Tensor *mask, float p) {
    assert(mask != nullptr);
    assert(mask->get_dim() == 1);
    auto length = mask->length();
    for (int i = 0; i < length; ++i) {
        static_cast<float*>(mask->get_data())[i] = dis(gen) < p ? 0 : 1;
    }
}

void CPUOps::calcAllGradNorm(const std::vector<Tensor*> &grads, Tensor *norm) {
    float tmp = 0;
    for (auto grad : grads) {
        float *data = static_cast<float*>(grad->get_data());
        for (int i = 0; i < grad->length(); ++i) {
            float &value = data[i];
            tmp += std::pow(value, 2);
        }
    }
    assert(norm->get_shape().size() == 1);
    assert(norm->get_shape()[0] == 1);
    float *norm_data = static_cast<float*>(norm->get_data());
    norm_data[0] = tmp;
}

void CPUOps::clipGrad(Tensor *grad, const Tensor *norm, float grad_clip_val) {
    assert(grad != nullptr);
    assert(norm != nullptr);
    assert(norm->get_shape().size() == 1);
    assert(norm->get_shape()[0] == 1);
    float *data = static_cast<float*>(grad->get_data());
    float *norm_data = static_cast<float*>(norm->get_data());
    float norm_value = std::sqrt(norm_data[0]);
    if (norm_value > grad_clip_val) {
        for (int i = 0; i < grad->length(); ++i) {
            data[i] *= grad_clip_val / norm_value;
        }
    }
}

void CPUOps::adamStep(Tensor *w, Tensor *grad, Tensor *m, Tensor *v, int t, float lr, float beta1, float beta2, float epsilon) {
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

    float *w_data = static_cast<float*>(w->get_data());
    float *grad_data = static_cast<float*>(grad->get_data());
    float *m_data = static_cast<float*>(m->get_data());
    float *v_data = static_cast<float*>(v->get_data());
    for (int i = 0; i < w->length(); ++i) {
        m_data[i] = beta1 * m_data[i] + (1 - beta1) * grad_data[i];
        v_data[i] = beta2 * v_data[i] + (1 - beta2) * std::pow(grad_data[i], 2);
        float m_hat = m_data[i] / (1 - std::pow(beta1, t));
        float v_hat = v_data[i] / (1 - std::pow(beta2, t));
        w_data[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}

void CPUOps::init_weight_gauss(Tensor *tensor, float mean, float sigma) {
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w(seed1);
    std::normal_distribution<float> distribution_w(0.0, sigma);
    float *data = static_cast<float*>(tensor->get_data());
    for (int i = 0; i < tensor->length(); ++i) {
        data[i] = distribution_w(generator_w) + mean;
    }
}

void CPUOps::init_weight_uniform(Tensor *tensor, float sigma) {
    
}

void CPUOps::init_weight_for_dbg(Tensor *tensor, float scale) {
    assert(tensor != nullptr);
    assert(tensor->get_data() != nullptr);
    assert(tensor->length() > 0);

    if (tensor->get_dtype() == FLOAT32) {
        float *data = static_cast<float*>(tensor->get_data());
        for (int i = 0; i < tensor->length(); ++i) {
            data[i] = static_cast<float>(i) * 1e-5 * scale;
        }
    } else if (tensor->get_dtype() == INT32) {
        int32_t *data = static_cast<int32_t*>(tensor->get_data());
        for (int i = 0; i < tensor->length(); ++i) {
            data[i] = i % 10;
        }
    } else {
        assert(false);
    }
    
}

void CPUOps::fill(Tensor *tensor, float value) {
    assert(tensor != nullptr);
    assert(tensor->get_data() != nullptr);
    assert(tensor->length() > 0);

    float *data = static_cast<float*>(tensor->get_data());
    for (int i = 0; i < tensor->length(); ++i) {
        data[i] = value;
    }   
}

void CPUOps::reshape_deep_cp(
    Tensor *dst_tensor, const Tensor *src_tensor,
    const Tensor *src_shape, const Tensor *src_strides) {
    
    assert(dst_tensor->get_dtype() == src_tensor->get_dtype());
    assert(
        dst_tensor->get_dtype() == INT32 ||
        dst_tensor->get_dtype() == FLOAT32
    );

    auto dtype = dst_tensor->get_dtype();
    auto src_shape_data = static_cast<int32_t*>(src_shape->get_data());
    auto src_strides_data = static_cast<int32_t*>(src_strides->get_data());
    auto dim = src_tensor->get_dim();
    auto length = src_tensor->length();

    if (dtype == INT32) {
        assert(false);
    } else if (dtype == FLOAT32) {
        auto dst_data = static_cast<float*>(dst_tensor->get_data());
        auto src_data = static_cast<float*>(src_tensor->get_data());
        for (int i = 0; i < length; ++i) {
            int offset = 0;
            int index = i;
            auto tmp_length = length;
            for (int j = 0; j < dim; ++j) {
                tmp_length /= src_shape_data[j];
                auto cur_dim_index = index / tmp_length;
                offset += cur_dim_index * src_strides_data[j];
                index %= tmp_length;
            }
            dst_data[i] = src_data[offset];
        }
    } else {
        assert(false);
    }
}

void CPUOps::repeat_interleave(Tensor *lhs, Tensor *res, int n) {
    assert(lhs->get_dtype() == INT32);
    assert(res->get_dtype() == INT32);
    assert(lhs != nullptr);
    assert(res != nullptr);

    assert(lhs->get_dim() == 1);
    assert(res->get_dim() == 1);

    auto l_length = lhs->length();
    auto r_length = res->length();

    assert(l_length * n == r_length);

    for (int i = 0; i < l_length; ++i) {
        for (int j = 0; j < n; ++j) {
            static_cast<int32_t*>(res->get_data())[i * n + j] = 
                static_cast<int32_t*>(lhs->get_data())[i];
        }
    }
}

void CPUOps::sequence_mask(Tensor *lhs, const Tensor *mask, Tensor *res, float value) {
    assert(lhs != nullptr);
    assert(mask != nullptr);
    assert(res != nullptr);

    assert(lhs->get_dtype() == FLOAT32);
    assert(mask->get_dtype() == INT32);
    assert(res->get_dtype() == FLOAT32);

    assert(lhs->get_dim() == 2);
    assert(mask->get_dim() == 1);
    assert(res->get_dim() == 2);

    auto lshape = lhs->get_shape();
    auto mshape = mask->get_shape();
    auto rshape = res->get_shape();

    assert(lshape[0] == mshape[0]);
    assert(lshape[1] == rshape[1]);
    assert(rshape[0] == mshape[0]);

    auto lstrides = lhs->get_strides();
    auto mstrides = mask->get_strides();
    auto rstrides = res->get_strides();

    for (int i = 0; i < lshape[0]; ++i) {
        for (int j = 0; j < lshape[1]; ++j) {
            static_cast<float*>(res->get_data())[i * rstrides[0] + j * rstrides[1]] = 
                static_cast<int32_t*>(mask->get_data())[i * mstrides[0]] <= j ? value : 
                static_cast<float*>(lhs->get_data())[i * lstrides[0] + j * lstrides[1]];
        }
    }
}

void CPUOps::softmax(Tensor *lhs, Tensor *res) {
    auto l_shape = lhs->get_shape();
    auto r_shape = res->get_shape();
    assert(l_shape == r_shape);
    assert(lhs->get_dtype() == FLOAT32);
    assert(res->get_dtype() == FLOAT32);
    assert(lhs->get_dim() == 3);
    assert(res->get_dim() == 3);
    auto lstrides = lhs->get_strides();
    auto rstrides = res->get_strides();
    for (int i = 0; i < l_shape[0]; ++i) {
        for (int j = 0; j < l_shape[1]; ++j) {
            float max = static_cast<float*>(lhs->get_data())[i * lstrides[0] + j * lstrides[1]];
            for (int k = 0; k < l_shape[2]; ++k) {
                auto e = static_cast<float*>(lhs->get_data())[i * lstrides[0] + j * lstrides[1] + k * lstrides[2]];
                if (max < e) {
                    max = e;
                }
            }
            float sum = 0;
            for (int k = 0; k < l_shape[2]; ++k) {
                float e = static_cast<float*>(lhs->get_data())[i * lstrides[0] + j * lstrides[1] + k * lstrides[2]];
                e = std::exp(e - max);
                sum += e;
            }
            for (int k = 0; k < l_shape[2]; ++k) {
                static_cast<float*>(res->get_data())[i * rstrides[0] + j * rstrides[1] + k * rstrides[2]] =
                    std::exp(static_cast<float*>(lhs->get_data())[i * lstrides[0] + j * lstrides[1] + k * lstrides[2]] - max) / sum;
            }
        }
    }
}

void CPUOps::softmax_bacward(Tensor *target_grad, const Tensor *softmax_res, Tensor *grad) {
    assert(target_grad != nullptr);
    assert(softmax_res != nullptr);
    assert(grad != nullptr);

    assert(target_grad->get_dtype() == FLOAT32);
    assert(softmax_res->get_dtype() == FLOAT32);
    assert(grad->get_dtype() == FLOAT32);

    assert(target_grad->get_dim() == 3);
    assert(softmax_res->get_dim() == 3);
    assert(grad->get_dim() == 3);

    auto t_shape = target_grad->get_shape();
    auto s_shape = softmax_res->get_shape();
    auto g_shape = grad->get_shape();

    assert(t_shape == s_shape);
    assert(t_shape == g_shape);

    auto t_strides = target_grad->get_strides();
    auto s_strides = softmax_res->get_strides();
    auto g_strides = grad->get_strides();

    float *target_grad_data = static_cast<float*>(target_grad->get_data());
    float *softmax_res_data = static_cast<float*>(softmax_res->get_data());
    float *grad_data = static_cast<float*>(grad->get_data());

    for (int i = 0; i < t_shape[0]; ++i) {
        for (int j = 0; j < t_shape[1]; ++j) {
            for (int target = 0; target < t_shape[2]; ++target) {
                for (int k = 0; k < t_shape[2]; ++k) {
                    auto tg_target_pos = i * t_strides[0] + j * t_strides[1] + target * t_strides[2];
                    // auto tg_k_pos = i * t_strides[0] + j * t_strides[1] + k * t_strides[2];
                    auto sm_target_pos = i * s_strides[0] + j * s_strides[1] + target * s_strides[2];
                    auto sm_k_pos = i * s_strides[0] + j * s_strides[1] + k * s_strides[2];
                    // auto g_target_pos = i * g_strides[0] + j * g_strides[1] + target * g_strides[2];
                    auto g_k_pos = i * g_strides[0] + j * g_strides[1] + k * g_strides[2];
                    auto softmax_res_k = softmax_res_data[sm_k_pos];
                    auto softmax_res_target = softmax_res_data[sm_target_pos];
                    auto grad_k = grad_data[g_k_pos];
                    if (target == k) {
                        target_grad_data[tg_target_pos] += 
                            softmax_res_k * (1 - softmax_res_k) * grad_k;
                    } else {
                        target_grad_data[tg_target_pos] += 
                            -softmax_res_target * softmax_res_k * grad_k;
                    }
                }
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

void CPUOps::cp_device_to_device(void* dst, const void* src, size_t size) {
    ::memcpy(dst, src, size);
}

void CPUOps::free(void* ptr) {
    ::free(ptr);
}

void CPUOps::cp_to_device(Tensor *dst_tensor, char *src, size_t size) {
    assert(dst_tensor != nullptr);
    assert(src != nullptr);
    assert(size > 0);
    assert(dst_tensor->get_data() != nullptr);
    assert(dst_tensor->size() == size);
    memcpy(dst_tensor->get_data(), src, size);
}

void CPUOps::cp_from_device(char *dst, const Tensor *src_tensor, size_t size) {
    assert(dst != nullptr);
    assert(src_tensor != nullptr);
    assert(size > 0);
    assert(src_tensor->get_data() != nullptr);
    memcpy(dst, src_tensor->get_data(), size);
}