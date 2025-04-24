#ifndef CPU_OPS_CUH
#define CPU_OPS_CUH

#include "backends/ops.cuh"

class CPUBackendOps : public BackendOps {
    public:
        void cp_to_device(void* dst, const void* src, size_t size) override;
        void cp_from_device(void* dst, const void* src, size_t size) override;
};

#endif