#ifndef OPS_CUH
#define OPS_CUH

#include "autograd/node.cuh"
class BackendOps {
    public:
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
};

extern BackendOps *g_backend_ops;
#endif