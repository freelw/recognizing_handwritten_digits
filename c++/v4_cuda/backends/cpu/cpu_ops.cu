#include "cpu_ops.cuh"


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