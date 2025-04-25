#ifndef CPU_OPS_CUH
#define CPU_OPS_CUH

#include "backends/ops.cuh"

class CPUBackendOps : public BackendOps {
    public:
        virtual bool is_gpu() override;
        virtual void cp_to_device(void* dst, const void* src, size_t size) override;
        virtual void cp_from_device(void* dst, const void* src, size_t size) override;
        virtual Matrix *CrossEntropyLoss(
            Matrix *input,
            const std::vector<uint> &labels,
            std::vector<autograd_cuda::CrosEntropyInfo> &info) override;
        virtual Matrix *CrossEntropyLossMask(
            Matrix *input,
            const std::vector<uint> &labels,
            std::vector<autograd_cuda::CrosEntropyInfo> &info,
            const std::vector<bool> &mask) override;
        virtual Matrix *Norm(Matrix *w,
            const std::vector<DATATYPE> &avg_res,
            const std::vector<DATATYPE> &var_res,
            DATATYPE eps) override;
        virtual Matrix *Softmax(Matrix *w) override;
        virtual std::vector<Matrix*> split0(Matrix *w) override;
        virtual std::vector<Matrix*> split1(Matrix *w, uint step) override;
        virtual void CrossEntropyEdgeBackward(
            Matrix *w,
            Matrix *grad,
            const std::vector<uint> &labels,
            const std::vector<autograd_cuda::CrosEntropyInfo> &info) override;
        virtual void CrossEntropyMaskEdgeBackward(
            Matrix *w,
            Matrix *grad,
            const std::vector<uint> &labels,
            const std::vector<autograd_cuda::CrosEntropyInfo> &info,
            const std::vector<bool> &mask) override;
        virtual void NormEdgeBackward(
            Matrix *w,
            Matrix *grad,
            const std::vector<DATATYPE> &avg_res,
            const std::vector<DATATYPE> &var_res,
            DATATYPE eps) override;
        virtual DATATYPE *allocDeviceMem(size_t size) override;
        virtual void releaseDeviceMem(DATATYPE *ptr) override;
        virtual void expand_add(Matrix *w, const Matrix &m) override;
        virtual void operator_add(Matrix *w, const Matrix &m) override;
        virtual void pow2(Matrix *w) override;
};
#endif