#include "cuda_ops.cuh"


bool GPUBackendOps::is_gpu() {
    return true;
}

// set all interface un implemented
void GPUBackendOps::cp_to_device(void* dst, const void* src, size_t size) {
    std::cerr << "cp_to_device unimplemented" << std::endl;
    assert(false);
}

void GPUBackendOps::cp_from_device(void* dst, const void* src, size_t size) {
    std::cerr << "cp_from_device unimplemented" << std::endl;
    assert(false);
}


// virtual Matrix *CrossEntropyLoss(
//     Matrix *input,
//     const std::vector<uint> &labels,
//     std::vector<autograd_cuda::CrosEntropyInfo> &info) override;
Matrix *GPUBackendOps::CrossEntropyLoss(
    Matrix *input,
    const std::vector<uint> &labels,
    std::vector<autograd_cuda::CrosEntropyInfo> &info) {
    std::cerr << "CrossEntropyLoss unimplemented" << std::endl;
    assert(false);
    return nullptr;
}

// virtual Matrix *CrossEntropyLossMask(
//     Matrix *input,
//     const std::vector<uint> &labels,
//     std::vector<autograd_cuda::CrosEntropyInfo> &info,
//     const std::vector<bool> &mask) override;
Matrix *GPUBackendOps::CrossEntropyLossMask(
    Matrix *input,
    const std::vector<uint> &labels,
    std::vector<autograd_cuda::CrosEntropyInfo> &info,
    const std::vector<bool> &mask) {
    std::cerr << "CrossEntropyLossMask unimplemented" << std::endl;
    assert(false);
    return nullptr;
}

// virtual Matrix *Norm(Matrix *w,
//     const std::vector<DATATYPE> &avg_res,
//     const std::vector<DATATYPE> &var_res,
//     DATATYPE eps) override;
Matrix *GPUBackendOps::Norm(
    Matrix *w,
    const std::vector<DATATYPE> &avg_res,
    const std::vector<DATATYPE> &var_res,
    DATATYPE eps) {
    std::cerr << "Norm unimplemented" << std::endl;
    assert(false);
    return nullptr;
}

// virtual Matrix *Softmax(Matrix *w) override;
Matrix *GPUBackendOps::Softmax(Matrix *w) {
    std::cerr << "Softmax unimplemented" << std::endl;
    assert(false);
    return nullptr;
}

// virtual std::vector<Matrix*> split0(Matrix *w) override;
std::vector<Matrix*> GPUBackendOps::split0(Matrix *w) {
    std::cerr << "split0 unimplemented" << std::endl;
    assert(false);
    return std::vector<Matrix*>();
}

// virtual std::vector<Matrix*> split1(Matrix *w, uint step) override;
std::vector<Matrix*> GPUBackendOps::split1(Matrix *w, uint step) {
    std::cerr << "split1 unimplemented" << std::endl;
    assert(false);
    return std::vector<Matrix*>();
}

// virtual void CrossEntropyEdgeBackward(
//     Matrix *w,
//     Matrix *grad,
//     const std::vector<uint> &labels,
//     const std::vector<autograd_cuda::CrosEntropyInfo> &info) override;
void GPUBackendOps::CrossEntropyEdgeBackward(
    Matrix *w,
    Matrix *grad,
    const std::vector<uint> &labels,
    const std::vector<autograd_cuda::CrosEntropyInfo> &info) {
    std::cerr << "CrossEntropyEdgeBackward unimplemented" << std::endl;
    assert(false);
}

// virtual void CrossEntropyMaskEdgeBackward(
//     Matrix *w,
//     Matrix *grad,
//     const std::vector<uint> &labels,
//     const std::vector<autograd_cuda::CrosEntropyInfo> &info,
//     const std::vector<bool> &mask) override;
void GPUBackendOps::CrossEntropyMaskEdgeBackward(
    Matrix *w,
    Matrix *grad,
    const std::vector<uint> &labels,
    const std::vector<autograd_cuda::CrosEntropyInfo> &info,
    const std::vector<bool> &mask) {
    std::cerr << "CrossEntropyMaskEdgeBackward unimplemented" << std::endl;
    assert(false);
}

// virtual void NormEdgeBackward(
//     Matrix *w,
//     Matrix *grad,
//     const std::vector<DATATYPE> &avg_res,
//     const std::vector<DATATYPE> &var_res,
//     DATATYPE eps) override;
void GPUBackendOps::NormEdgeBackward(
    Matrix *w,
    Matrix *grad,
    const std::vector<DATATYPE> &avg_res,
    const std::vector<DATATYPE> &var_res,
    DATATYPE eps) {
    std::cerr << "NormEdgeBackward unimplemented" << std::endl;
    assert(false);
}

// virtual DATATYPE *allocDeviceMem(size_t size) override;
DATATYPE *GPUBackendOps::allocDeviceMem(size_t size) {
    std::cerr << "allocDeviceMem unimplemented" << std::endl;
    assert(false);
    return nullptr;
}

// virtual void releaseDeviceMem(DATATYPE *ptr) override;
void GPUBackendOps::releaseDeviceMem(DATATYPE *ptr) {
    std::cerr << "releaseDeviceMem unimplemented" << std::endl;
    assert(false);
}

// virtual void expand_add(Matrix *w, const Matrix &m) override;
void GPUBackendOps::expand_add(Matrix *w, const Matrix &m) {
    std::cerr << "expand_add unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_add(Matrix *w, const Matrix &m) override;
void GPUBackendOps::operator_add(Matrix *w, const Matrix &m) {
    std::cerr << "operator_add unimplemented" << std::endl;
    assert(false);
}

// virtual void pow2(Matrix *w) override;
void GPUBackendOps::pow2(Matrix *w) {
    std::cerr << "pow2 unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_add_val(Matrix *w, DATATYPE v) override;
void GPUBackendOps::operator_add_val(Matrix *w, DATATYPE v) {
    std::cerr << "operator_add_val unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_minus_val(Matrix *w, DATATYPE v) override;
void GPUBackendOps::operator_minus_val(Matrix *w, DATATYPE v) {
    std::cerr << "operator_minus_val unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_negative(Matrix *w) override;
void GPUBackendOps::operator_negative(Matrix *w) {
    std::cerr << "operator_negative unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_val_minus(DATATYPE v, Matrix *w) override;
void GPUBackendOps::operator_val_minus(DATATYPE v, Matrix *w) {
    std::cerr << "operator_val_minus unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_minus(Matrix *w, const Matrix &m) override;
void GPUBackendOps::operator_minus(Matrix *w, const Matrix &m) {
    std::cerr << "operator_minus unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_multiply(Matrix *w, const Matrix &m) override;
void GPUBackendOps::operator_multiply(Matrix *w, const Matrix &m) {
    std::cerr << "operator_multiply unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_multiply_val(Matrix *w, DATATYPE v) override;
void GPUBackendOps::operator_multiply_val(Matrix *w, DATATYPE v) {
    std::cerr << "operator_multiply_val unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_divide(Matrix *w, const Matrix &m) override;
void GPUBackendOps::operator_divide(Matrix *w, const Matrix &m) {
    std::cerr << "operator_divide unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_divide_val(Matrix *w, DATATYPE v) override;
void GPUBackendOps::operator_divide_val(Matrix *w, DATATYPE v) {
    std::cerr << "operator_divide_val unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_relu(Matrix *w) override;
void GPUBackendOps::operator_relu(Matrix *w) {
    std::cerr << "operator_relu unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_relu_prime(Matrix *w) override;
void GPUBackendOps::operator_relu_prime(Matrix *w) {
    std::cerr << "operator_relu_prime unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_tanh(Matrix *w) override;
void GPUBackendOps::operator_tanh(Matrix *w) {
    std::cerr << "operator_tanh unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_tanh_prime(Matrix *w) override;
void GPUBackendOps::operator_tanh_prime(Matrix *w) {
    std::cerr << "operator_tanh_prime unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_equal(Matrix *w, const Matrix &m) override;
void GPUBackendOps::operator_equal(Matrix *w, const Matrix &m) {
    std::cerr << "operator_equal unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_at(Matrix *res, Matrix *w, const Matrix &m) override;
void GPUBackendOps::operator_at(Matrix *res, Matrix *w, const Matrix &m) {
    std::cerr << "operator_at unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_transpose(Matrix *res, Matrix *w) override;
void GPUBackendOps::operator_transpose(Matrix *res, Matrix *w) {
    std::cerr << "operator_transpose unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_assign(Matrix *res, Matrix *w) override;
void GPUBackendOps::operator_assign(Matrix *res, Matrix *w) {
    std::cerr << "operator_assign unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_sum(Matrix *res, Matrix *w) override;
void GPUBackendOps::operator_sum(Matrix *res, Matrix *w) {
    std::cerr << "operator_sum unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_split(std::vector<Matrix *> &res, Matrix *w) override;
void GPUBackendOps::operator_split(std::vector<Matrix *> &res, Matrix *w) {
    std::cerr << "operator_split unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_fill(Matrix *w, DATATYPE value) override;
void GPUBackendOps::operator_fill(Matrix *w, DATATYPE value) {
    std::cerr << "operator_fill unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_argMax(std::vector<uint> &res, Matrix *w) override;
void GPUBackendOps::operator_argMax(std::vector<uint> &res, Matrix *w) {
    std::cerr << "operator_argMax unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_avg(std::vector<DATATYPE> &res, Matrix *w) override;
void GPUBackendOps::operator_avg(std::vector<DATATYPE> &res, Matrix *w) {
    std::cerr << "operator_avg unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_var(std::vector<DATATYPE> &res, Matrix *w) override;
void GPUBackendOps::operator_var(std::vector<DATATYPE> &res, Matrix *w) {
    std::cerr << "operator_var unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_sigmoid(Matrix *w) override;
void GPUBackendOps::operator_sigmoid(Matrix *w) {
    std::cerr << "operator_sigmoid unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_sigmoid_prime(Matrix *w) override;
void GPUBackendOps::operator_sigmoid_prime(Matrix *w) {
    std::cerr << "operator_sigmoid_prime unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_init_weight(Matrix *w, DATATYPE sigma, DATATYPE mean = 0) override;
void GPUBackendOps::operator_init_weight(Matrix *w, DATATYPE sigma, DATATYPE mean) {
    std::cerr << "operator_init_weight unimplemented" << std::endl;
    assert(false);
}

// virtual void operator_init_weight_uniform(Matrix *w, DATATYPE sigma) override;
void GPUBackendOps::operator_init_weight_uniform(Matrix *w, DATATYPE sigma) {
    std::cerr << "operator_init_weight_uniform unimplemented" << std::endl;
    assert(false);
}
