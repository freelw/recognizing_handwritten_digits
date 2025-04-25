#include "cpu_ops.cuh"
#include <random>
#include <chrono>

bool CPUBackendOps::is_gpu() {
    return false;
}

void CPUBackendOps::cp_to_device(void* dst, const void* src, size_t size) {
    
}

void CPUBackendOps::cp_from_device(void* dst, const void* src, size_t size) {
    
}

Matrix *CPUBackendOps::CrossEntropyLoss(
    Matrix *input,
    const std::vector<uint> &labels,
    std::vector<autograd_cuda::CrosEntropyInfo> &info) {

    input->increase_cpu_ver();
    assert(input->getShape().colCnt == labels.size());
    assert(info.size() == 0);
    Matrix *loss = allocTmpMatrix(Shape(1,1));
    DATATYPE loss_value = 0;
    info.resize(input->getShape().colCnt);

    #pragma omp parallel for reduction(+:loss_value)
    for (uint j = 0; j < input->getShape().colCnt; ++ j) {
        DATATYPE max = (*input)[0][j];
        for (uint i = 0; i < input->getShape().rowCnt; ++ i) {
            auto e = (*input)[i][j];
            if (max < e) {
                max = e;
            }
        }
        DATATYPE sum = 0;
        auto target = labels[j];
        DATATYPE zt = (*input)[target][j];
        for (uint i = 0; i < input->getShape().rowCnt; ++ i) {
            DATATYPE e = (*input)[i][j];
            e = std::exp(e-max);
            sum += e;
        }
        autograd_cuda::CrosEntropyInfo &p = info[j];
        p.sum = sum;
        p.max = max;
        loss_value += -(zt - max - log(sum));
    }
    (*loss)[0][0] = loss_value/labels.size();
    loss->increase_cpu_ver();
    return loss;
}

Matrix *CPUBackendOps::CrossEntropyLossMask(
    Matrix *input,
    const std::vector<uint> &labels,
    std::vector<autograd_cuda::CrosEntropyInfo> &info,
    const std::vector<bool> &mask) {
    input->increase_cpu_ver();
    assert(input->getShape().colCnt == labels.size());
    assert(input->getShape().colCnt == mask.size());
    assert(info.size() == 0);

    Matrix *loss = allocTmpMatrix(Shape(1,1));
    DATATYPE loss_value = 0;
    info.resize(input->getShape().colCnt);
    uint mask_cnt = 0;
    // #pragma omp parallel for reduction(+:mask_cnt)
    for (uint i = 0; i < mask.size(); ++ i) {
        mask_cnt += mask[i];
    }
    if (mask_cnt == 0) {
        (*loss)[0][0] = 0;
        return loss;
    }

    // #pragma omp parallel for reduction(+:loss_value)
    for (uint j = 0; j < input->getShape().colCnt; ++ j) {
        if (!mask[j]) {
            continue;
        }
        DATATYPE max = (*input)[0][j];
        for (uint i = 0; i < input->getShape().rowCnt; ++ i) {
            auto e = (*input)[i][j];
            if (max < e) {
                max = e;
            }
        }
        DATATYPE sum = 0;
        auto target = labels[j];
        DATATYPE zt = (*input)[target][j];
        for (uint i = 0; i < input->getShape().rowCnt; ++ i) {
            DATATYPE e = (*input)[i][j];
            e = std::exp(e-max);
            sum += e;
        }
        autograd_cuda::CrosEntropyInfo &p = info[j];
        p.sum = sum;
        p.max = max;
        loss_value += -(zt - max - log(sum));
    }
    (*loss)[0][0] = loss_value/mask_cnt;
    loss->increase_cpu_ver();
    return loss;
}

Matrix *CPUBackendOps::Norm(
    Matrix *w,
    const std::vector<DATATYPE> &avg_res,
    const std::vector<DATATYPE> &var_res,
    DATATYPE eps) {
    auto *tmp = allocTmpMatrix(w);
    Shape shape = tmp->getShape();
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        for (uint j = 0; j < shape.colCnt; ++ j) {
            (*tmp)[i][j] = ((*w)[i][j] - avg_res[j]) / sqrt(var_res[j] + eps);
        }
    }
    tmp->increase_cpu_ver();
    return tmp;
}

Matrix *CPUBackendOps::Softmax(Matrix *w) {
    auto *tmp = allocTmpMatrix(w);
    Shape shape = tmp->getShape();
    for (uint j = 0; j < shape.colCnt; ++ j) {
        DATATYPE max = (*w)[0][j];
        for (uint i = 0; i < shape.rowCnt; ++ i) {
            if (max < (*w)[i][j]) {
                max = (*w)[i][j];
            }
        }
        DATATYPE sum = 0;
        for (uint i = 0; i < shape.rowCnt; ++ i) {
            DATATYPE e = std::exp((*w)[i][j] - max);
            sum += e;
            (*tmp)[i][j] = e;
        }
        for (uint i = 0; i < shape.rowCnt; ++ i) {
            (*tmp)[i][j] /= sum;
        }
    }
    tmp->increase_cpu_ver();
    return tmp;
}

std::vector<Matrix*> CPUBackendOps::split0(Matrix *w) {
    Shape shape = w->getShape();
    uint colCnt = shape.colCnt;
    uint rowCnt = shape.rowCnt;
    std::vector<Matrix*> ret;
    ret.reserve(colCnt);
    for (uint i = 0; i < colCnt; ++ i) {
        Matrix *m = allocTmpMatrix(Shape(rowCnt, 1));
        for (uint j = 0; j < rowCnt; ++ j) {
            (*m)[j][0] = (*w)[j][i];
        }
        ret.emplace_back(m);
        m->increase_cpu_ver();
    }
    return ret;
}

std::vector<Matrix*> CPUBackendOps::split1(Matrix *w, uint step) {
    Shape shape = w->getShape();
    uint colCnt = shape.colCnt;
    uint rowCnt = shape.rowCnt;
    assert(step > 0 && rowCnt % step == 0);
    std::vector<Matrix *> ret;
    for (uint i = 0; i < rowCnt; i += step) {
        Matrix *m = allocTmpMatrix(Shape(step, colCnt));
        for (uint j = 0; j < step; ++ j) {
            for (uint k = 0; k < colCnt; ++ k) {
                (*m)[j][k] = (*w)[i+j][k];
            }
        }
        m->increase_cpu_ver();
        ret.push_back(m);
    }
    return ret;
}

void CPUBackendOps::CrossEntropyEdgeBackward(
    Matrix *w,
    Matrix *grad,
    const std::vector<uint> &labels,
    const std::vector<autograd_cuda::CrosEntropyInfo> &info) {
    
    #pragma omp parallel for
    for (uint i = 0; i < labels.size(); ++i) {
        auto target = labels[i];
        DATATYPE max = info[i].max;
        DATATYPE sum = info[i].sum;
        for (uint j = 0; j < w->getShape().rowCnt; ++j) {
            if (j == target) {
                continue;
            }
            auto &_grad = (*grad)[j][i];
            _grad = std::exp((*w)[j][i] - max) / sum / labels.size();
        }
        (*grad)[target][i] = (std::exp((*w)[target][i] - max) / sum - 1) / labels.size();
    }
    grad->increase_cpu_ver();
}

void CPUBackendOps::CrossEntropyMaskEdgeBackward(
    Matrix *w,
    Matrix *grad,
    const std::vector<uint> &labels,
    const std::vector<autograd_cuda::CrosEntropyInfo> &info,
    const std::vector<bool> &mask) {
    uint mask_cnt = 0;
    #pragma omp parallel for reduction(+:mask_cnt)
    for (uint i = 0; i < mask.size(); ++ i) {
        mask_cnt += mask[i];
    }
    if (mask_cnt == 0) {
        return;
    }
    Shape shape = w->getShape();
    #pragma omp parallel for
    for (uint j = 0; j < shape.colCnt; ++ j) {
        if (!mask[j]) {
            continue;
        }
        auto target = labels[j];
        DATATYPE max = info[j].max;
        DATATYPE sum = info[j].sum;
        for (uint i = 0; i < shape.rowCnt; ++ i) {
            if (i == target) {
                continue;
            }
            auto &_grad = (*grad)[i][j];
            _grad = std::exp((*w)[i][j] - max) / sum / mask_cnt;
        }
        (*grad)[target][j] = (std::exp((*w)[target][j] - max) / sum - 1) / mask_cnt;
    }
    grad->increase_cpu_ver();
}

void CPUBackendOps::NormEdgeBackward(
    Matrix *w,
    Matrix *grad,
    const std::vector<DATATYPE> &avg_res,
    const std::vector<DATATYPE> &var_res,
    DATATYPE eps) {
    assert(false);
}

DATATYPE *CPUBackendOps::allocDeviceMem(size_t size) {
    return nullptr;
}

void CPUBackendOps::releaseDeviceMem(DATATYPE *ptr) {
    assert(ptr == nullptr);
}

void CPUBackendOps::expand_add(Matrix *w, const Matrix &m) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*w)[i][j] += m[i][0];
        }
    }
    w->increase_cpu_ver();
}

void CPUBackendOps::operator_add(Matrix *w, const Matrix &m) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*w)[i][j] += m[i][j];
        }
    }
    w->increase_cpu_ver();
}

void CPUBackendOps::pow2(Matrix *w) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            auto &r = (*w)[i][j];
            r = std::pow(r, 2);
        }
    }
    w->increase_cpu_ver();
}

void CPUBackendOps::operator_add_val(Matrix *w, DATATYPE v) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*w)[i][j] += v;
        }
    }
    w->increase_cpu_ver();
}

void CPUBackendOps::operator_minus_val(Matrix *w, DATATYPE v) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*w)[i][j] -= v;
        }
    }
    w->increase_cpu_ver();
}

void CPUBackendOps::operator_negative(Matrix *w) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*w)[i][j] = -(*w)[i][j];
        }
    }
    w->increase_cpu_ver();
}

void CPUBackendOps::operator_val_minus(DATATYPE v, Matrix *w) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*w)[i][j] = v - (*w)[i][j];
        }
    }
    w->increase_cpu_ver();
}

void CPUBackendOps::operator_minus(Matrix *w, const Matrix &m) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*w)[i][j] -= m[i][j];
        }
    }
    w->increase_cpu_ver();
}

void CPUBackendOps::operator_multiply(Matrix *w, const Matrix &m) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*w)[i][j] *= m[i][j];
        }
    }
}

void CPUBackendOps::operator_multiply_val(Matrix *w, DATATYPE v) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*w)[i][j] *= v;
        }
    }
    w->increase_cpu_ver();
}

void CPUBackendOps::operator_divide(Matrix *w, const Matrix &m) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*w)[i][j] /= m[i][j];
        }
    }
    w->increase_cpu_ver();
}

void CPUBackendOps::operator_divide_val(Matrix *w, DATATYPE v) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*w)[i][j] /= v;
        }
    }
    w->increase_cpu_ver();
}

void CPUBackendOps::operator_relu(Matrix *w) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            auto &r = (*w)[i][j];
            r = r > 0 ? r : 0;
        }
    }
    w->increase_cpu_ver();
}

void CPUBackendOps::operator_relu_prime(Matrix *w) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            auto &r = (*w)[i][j];
            r = r > 0 ? 1 : 0;
        }
    }
    w->increase_cpu_ver();
}

void CPUBackendOps::operator_tanh(Matrix *w) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            auto &r = (*w)[i][j];
            r = std::tanh(r);
        }
    }
    w->increase_cpu_ver();
}

void CPUBackendOps::operator_tanh_prime(Matrix *w) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            auto &r = (*w)[i][j];
            r = 1 - std::pow(std::tanh(r), 2);
        }
    }
    w->increase_cpu_ver();
}

void CPUBackendOps::operator_equal(Matrix *w, const Matrix &m) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*w)[i][j] = m[i][j];
        }
    }
    w->increase_cpu_ver();
}

void CPUBackendOps::operator_at(Matrix *res, Matrix *w, Matrix &m) {

    auto shape = w->getShape();
    auto mshape = m.getShape();
    DATATYPE *A = w->getLowLevelData();
    DATATYPE *B = m.getLowLevelData();
    DATATYPE *C = res->getLowLevelData();
    #pragma omp parallel for num_threads(OMP_THREADS)
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint k = 0; k < shape.colCnt; ++k) {
            for (uint j = 0; j < mshape.colCnt; ++j) {
                C[i * mshape.colCnt + j] += A[i * shape.colCnt + k] * B[k * mshape.colCnt + j];
            }
        }
    }
    res->increase_cpu_ver();
}

void CPUBackendOps::operator_transpose(Matrix *res, Matrix *w) {
    auto shape = w->getShape();
    auto rshape = res->getShape();
    assert(shape.rowCnt == rshape.colCnt);
    assert(shape.colCnt == rshape.rowCnt);
    #pragma omp parallel for
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*res)[j][i] = (*w)[i][j];
        }
    }
    res->increase_cpu_ver();
}

void CPUBackendOps::operator_assign(Matrix *w, Matrix *m) {
    auto shape = w->getShape();
    assert(shape == m->getShape());
    memcpy(w->data, m->data, sizeof(DATATYPE) * shape.size());
    w->increase_cpu_ver();
}

void CPUBackendOps::operator_sum(Matrix *res, Matrix *w) {
    auto shape = w->getShape();
    auto rshape = res->getShape();
    assert(shape.rowCnt == rshape.rowCnt);
    assert(rshape.colCnt == 1);
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        for (uint j = 0; j < shape.colCnt; ++ j) {
            (*res)[i][0] += (*w)[i][j];
        }
    }
    res->increase_cpu_ver();
}

void CPUBackendOps::operator_split(std::vector<Matrix *> &res, Matrix *w) {
    auto shape = w->getShape();
    uint colCnt = shape.colCnt;
    uint rowCnt = shape.rowCnt;
    assert(res.size() == colCnt);
    for (uint i = 0; i < colCnt; ++ i) {
        for (uint j = 0; j < rowCnt; ++ j) {
            (*res[i])[j][0] = (*w)[j][i];
        }
        res[i]->increase_cpu_ver();
    }
}

void CPUBackendOps::operator_fill(Matrix *w, DATATYPE value) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        for (uint j = 0; j < shape.colCnt; ++ j) {
            (*w)[i][j] = value;
        }
    }
    w->increase_cpu_ver();
}

void CPUBackendOps::operator_argMax(std::vector<uint> &res, Matrix *w) {
    auto shape = w->getShape();
    res.resize(shape.colCnt);
    for (uint i = 0; i < shape.colCnt; ++ i) {
        DATATYPE max = (*w)[0][i];
        uint index = 0;
        for (uint j = 1; j < shape.rowCnt; ++ j) {
            if (max < (*w)[j][i]) {
                max = (*w)[j][i];
                index = j;
            }
        }
        res[i] = index;
    }
}

void CPUBackendOps::operator_avg(std::vector<DATATYPE> &res, Matrix *w) {
    auto shape = w->getShape();
    res.resize(shape.colCnt);
    for (uint i = 0; i < shape.colCnt; ++ i) {
        DATATYPE sum = 0;
        for (uint j = 0; j < shape.rowCnt; ++ j) {
            sum += (*w)[j][i];
        }
        res[i] = sum / shape.rowCnt;
    }
}

void CPUBackendOps::operator_var(std::vector<DATATYPE> &res, Matrix *w) {
    auto shape = w->getShape();
    res.resize(shape.colCnt);
    for (uint i = 0; i < shape.colCnt; ++ i) {
        DATATYPE sum = 0;
        DATATYPE avg = 0;
        for (uint j = 0; j < shape.rowCnt; ++ j) {
            sum += (*w)[j][i];
        }
        avg = sum / shape.rowCnt;
        sum = 0;
        for (uint j = 0; j < shape.rowCnt; ++ j) {
            DATATYPE e = (*w)[j][i] - avg;
            sum += std::pow(e, 2);
        }
        res[i] = sum / shape.rowCnt;
    }
}

DATATYPE _sigmoid(DATATYPE z) {
    return 1./(1.+exp(-z));
}

void CPUBackendOps::operator_sigmoid(Matrix *w) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        for (uint j = 0; j < shape.colCnt; ++ j) {
            (*w)[i][j] = _sigmoid((*w)[i][j]);
        }
    }
    w->increase_cpu_ver();
}

void CPUBackendOps::operator_sigmoid_prime(Matrix *w) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        for (uint j = 0; j < shape.colCnt; ++ j) {
            auto r = _sigmoid((*w)[i][j]);
            r = r * (1 - r);
        }
    }
    w->increase_cpu_ver();
}

void CPUBackendOps::operator_init_weight(Matrix *w, DATATYPE sigma, DATATYPE mean) {
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w(seed1);
    std::normal_distribution<DATATYPE> distribution_w(0.0, sigma);
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        for (uint j = 0; j < shape.colCnt; ++ j) {
            (*w)[i][j] = distribution_w(generator_w) + mean;
        }
    }
    w->increase_cpu_ver();
}

void CPUBackendOps::operator_init_weight_uniform(Matrix *w, DATATYPE sigma) {
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w(seed1);
    std::uniform_real_distribution<DATATYPE> distribution_w(-sigma, sigma);
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        for (uint j = 0; j < shape.colCnt; ++ j) {
            (*w)[i][j] = distribution_w(generator_w);
        }
    }
    w->increase_cpu_ver();
}