#ifndef OPS_CUH
#define OPS_CUH

#include "autograd/node.cuh"
class BackendOps {
    public:
        virtual bool is_gpu() = 0;
        virtual void cp_to_device(void* dst, const void* src, size_t size) = 0;
        virtual void cp_from_device(void* dst, const void* src, size_t size) = 0;
        virtual Matrix *CrossEntropyLoss(
            Matrix *input,
            const std::vector<uint> &labels,
            std::vector<autograd_cuda::CrosEntropyInfo> &info
        ) = 0;
        virtual Matrix *CrossEntropyLossMask(
            Matrix *input,
            const std::vector<uint> &labels,
            std::vector<autograd_cuda::CrosEntropyInfo> &info,
            const std::vector<bool> &mask) = 0;
        virtual Matrix *Norm(Matrix *w,
            const std::vector<DATATYPE> &avg_res,
            const std::vector<DATATYPE> &var_res,
            DATATYPE eps) = 0;
        virtual Matrix *Softmax(Matrix *w) = 0;
        virtual std::vector<Matrix*> split0(Matrix *w) = 0;
        virtual std::vector<Matrix*> split1(Matrix *w, uint step) = 0;
        virtual void CrossEntropyEdgeBackward(
            Matrix *w,
            Matrix *grad,
            const std::vector<uint> &labels,
            const std::vector<autograd_cuda::CrosEntropyInfo> &info) = 0;
        virtual void CrossEntropyMaskEdgeBackward(
            Matrix *w,
            Matrix *grad,
            const std::vector<uint> &labels,
            const std::vector<autograd_cuda::CrosEntropyInfo> &info,
            const std::vector<bool> &mask) = 0;
        virtual void NormEdgeBackward(
            Matrix *w,
            Matrix *grad,
            const std::vector<DATATYPE> &avg_res,
            const std::vector<DATATYPE> &var_res,
            DATATYPE eps) = 0;
        virtual DATATYPE *allocDeviceMem(size_t size) = 0;
        virtual void releaseDeviceMem(DATATYPE *ptr) = 0;
        virtual void expand_add(Matrix *w, const Matrix &m) = 0;
        virtual void operator_add(Matrix *w, const Matrix &m) = 0;
        virtual void pow2(Matrix *w) = 0;
        virtual void operator_add_val(Matrix *w, DATATYPE v) = 0;
        virtual void operator_minus_val(Matrix *w, DATATYPE v) = 0;
        virtual void operator_negative(Matrix *w) = 0;
        virtual void operator_val_minus(DATATYPE v, Matrix *w) = 0;
        virtual void operator_minus(Matrix *w, const Matrix &m) = 0;
        virtual void operator_multiply(Matrix *w, const Matrix &m) = 0;
        virtual void operator_multiply_val(Matrix *w, DATATYPE v) = 0;
        virtual void operator_divide(Matrix *w, const Matrix &m) = 0;
        virtual void operator_divide_val(Matrix *w, DATATYPE v) = 0;
        virtual void Relu(Matrix *w) = 0;
        virtual void Relu_prime(Matrix *w) = 0;
        virtual void tanh(Matrix *w) = 0;
        virtual void tanh_prime(Matrix *w) = 0;
        virtual void operator_equal(Matrix *w, const Matrix &m) = 0;
        virtual void operator_at(Matrix *res, Matrix *w, const Matrix &m) = 0;
};

extern BackendOps *g_backend_ops;
#endif