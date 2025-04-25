#include "cpu_ops.cuh"

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
    return loss;
}

Matrix *CPUBackendOps::CrossEntropyLossMask(
    Matrix *input,
    const std::vector<uint> &labels,
    std::vector<autograd_cuda::CrosEntropyInfo> &info,
    const std::vector<bool> &mask) {
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
}

void CPUBackendOps::operator_add(Matrix *w, const Matrix &m) {
    auto shape = w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*w)[i][j] += m[i][j];
        }
    }
}