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
};

extern BackendOps *g_backend_ops;
#endif