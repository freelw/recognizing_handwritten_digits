#ifndef OPS_CUH
#define OPS_CUH

class BackendOps {
    public:
        virtual void cp_to_device(void* dst, const void* src, size_t size) = 0;
        virtual void cp_from_device(void* dst, const void* src, size_t size) = 0;
};

extern BackendOps *g_backend_ops;
#endif