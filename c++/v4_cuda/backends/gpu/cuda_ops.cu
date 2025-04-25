#include "cuda_ops.cuh"
#include "kernel.cuh"


bool GPUBackendOps::is_gpu() {
    return true;
}

void GPUBackendOps::cp_to_device(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void GPUBackendOps::cp_from_device(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

Matrix *GPUBackendOps::CrossEntropyLoss(
    Matrix *input,
    const std::vector<uint> &labels,
    std::vector<autograd_cuda::CrosEntropyInfo> &info) {
    std::cerr << "CrossEntropyLoss unimplemented" << std::endl;
    assert(false);
    return nullptr;
}

Matrix *GPUBackendOps::CrossEntropyLossMask(
    Matrix *input,
    const std::vector<uint> &labels,
    std::vector<autograd_cuda::CrosEntropyInfo> &info,
    const std::vector<bool> &mask) {
    std::cerr << "CrossEntropyLossMask unimplemented" << std::endl;
    assert(false);
    return nullptr;
}

Matrix *GPUBackendOps::Norm(
    Matrix *w,
    const std::vector<DATATYPE> &avg_res,
    const std::vector<DATATYPE> &var_res,
    DATATYPE eps) {
    std::cerr << "Norm unimplemented" << std::endl;
    assert(false);
    return nullptr;
}

Matrix *GPUBackendOps::Softmax(Matrix *w) {
    std::cerr << "Softmax unimplemented" << std::endl;
    assert(false);
    return nullptr;
}

std::vector<Matrix*> GPUBackendOps::split0(Matrix *w) {
    std::cerr << "split0 unimplemented" << std::endl;
    assert(false);
    return std::vector<Matrix*>();
}

std::vector<Matrix*> GPUBackendOps::split1(Matrix *w, uint step) {
    std::cerr << "split1 unimplemented" << std::endl;
    assert(false);
    return std::vector<Matrix*>();
}

void GPUBackendOps::CrossEntropyEdgeBackward(
    Matrix *w,
    Matrix *grad,
    const std::vector<uint> &labels,
    const std::vector<autograd_cuda::CrosEntropyInfo> &info) {
    std::cerr << "CrossEntropyEdgeBackward unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::CrossEntropyMaskEdgeBackward(
    Matrix *w,
    Matrix *grad,
    const std::vector<uint> &labels,
    const std::vector<autograd_cuda::CrosEntropyInfo> &info,
    const std::vector<bool> &mask) {
    std::cerr << "CrossEntropyMaskEdgeBackward unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::NormEdgeBackward(
    Matrix *w,
    Matrix *grad,
    const std::vector<DATATYPE> &avg_res,
    const std::vector<DATATYPE> &var_res,
    DATATYPE eps) {
    std::cerr << "NormEdgeBackward unimplemented" << std::endl;
    assert(false);
}

DATATYPE *GPUBackendOps::allocDeviceMem(size_t size) {
    DATATYPE *ret = nullptr;
    cudaMalloc((void **)&ret, size);
    return ret;
}

void GPUBackendOps::releaseDeviceMem(DATATYPE *ptr) {
    assert(ptr != nullptr);
    cudaFree(ptr);
}

void GPUBackendOps::expand_add(Matrix *w, const Matrix &m) {
    std::cerr << "expand_add unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_add(Matrix *w, const Matrix &m) {
    std::cerr << "operator_add unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::pow2(Matrix *w) {
    std::cerr << "pow2 unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_add_val(Matrix *w, DATATYPE v) {
    std::cerr << "operator_add_val unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_minus_val(Matrix *w, DATATYPE v) {
    std::cerr << "operator_minus_val unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_negative(Matrix *w) {
    std::cerr << "operator_negative unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_val_minus(DATATYPE v, Matrix *w) {
    std::cerr << "operator_val_minus unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_minus(Matrix *w, const Matrix &m) {
    std::cerr << "operator_minus unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_multiply(Matrix *w, const Matrix &m) {
    std::cerr << "operator_multiply unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_multiply_val(Matrix *w, DATATYPE v) {
    std::cerr << "operator_multiply_val unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_divide(Matrix *w, const Matrix &m) {
    std::cerr << "operator_divide unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_divide_val(Matrix *w, DATATYPE v) {
    std::cerr << "operator_divide_val unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_relu(Matrix *w) {
    std::cerr << "operator_relu unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_relu_prime(Matrix *w) {
    std::cerr << "operator_relu_prime unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_tanh(Matrix *w) {
    std::cerr << "operator_tanh unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_tanh_prime(Matrix *w) {
    std::cerr << "operator_tanh_prime unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_equal(Matrix *w, const Matrix &m) {
    std::cerr << "operator_equal unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_at(Matrix *res, Matrix *w, Matrix &m) {
    w->sync();
    m.sync();

    auto wshape = w->getShape();
    auto mshape = m.getShape();
    auto rshape = res->getShape();

    assert(wshape.colCnt == mshape.rowCnt);
    assert(rshape.rowCnt == wshape.rowCnt);

    const int M = wshape.rowCnt;
    const int N = wshape.colCnt;
    const int P = mshape.colCnt;
    
    dim3 gridDim(
        (P + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH
    );
    dim3 blockDim(
        TILE_WIDTH,
        TILE_WIDTH
    );
    DATATYPE *d_Md = w->getLowLevelDataDevice();
    DATATYPE *d_Nd = m.getLowLevelDataDevice();
    DATATYPE *d_Pd = res->getLowLevelDataDevice();

    matrixmul<<<gridDim, blockDim>>>(d_Md, d_Nd, d_Pd, M, N, P);
    res->increase_gpu_ver();
}

void GPUBackendOps::operator_transpose(Matrix *res, Matrix *w) {
    std::cerr << "operator_transpose unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_assign(Matrix *res, Matrix *w) {
    std::cerr << "operator_assign unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_sum(Matrix *res, Matrix *w) {
    std::cerr << "operator_sum unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_split(std::vector<Matrix *> &res, Matrix *w) {
    std::cerr << "operator_split unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_fill(Matrix *w, DATATYPE value) {
    std::cerr << "operator_fill unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_argMax(std::vector<uint> &res, Matrix *w) {
    std::cerr << "operator_argMax unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_avg(std::vector<DATATYPE> &res, Matrix *w) {
    std::cerr << "operator_avg unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_var(std::vector<DATATYPE> &res, Matrix *w) {
    std::cerr << "operator_var unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_sigmoid(Matrix *w) {
    std::cerr << "operator_sigmoid unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_sigmoid_prime(Matrix *w) {
    std::cerr << "operator_sigmoid_prime unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_init_weight(Matrix *w, DATATYPE sigma, DATATYPE mean) {
    std::cerr << "operator_init_weight unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::operator_init_weight_uniform(Matrix *w, DATATYPE sigma) {
    std::cerr << "operator_init_weight_uniform unimplemented" << std::endl;
    assert(false);
}
