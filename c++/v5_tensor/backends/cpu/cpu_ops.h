#ifndef CPU_OPS_H
#define CPU_OPS_H

#include "backends/backend_ops.h"

class CPUOps : public BackendOps {
    public:
        CPUOps() = default;
        ~CPUOps() override = default;
        void add(Tensor *lhs, const Tensor *rhs, Tensor *res) override;
        void addEq(Tensor *lhs, const Tensor *rhs) override;
        void expandAdd(Tensor *lhs, const Tensor *rhs, Tensor *res) override;
        void at(Tensor *lhs, const Tensor *rhs, Tensor *res) override;
        void mul(Tensor *lhs, const Tensor *rhs, Tensor *res) override;
        void sum(Tensor *lhs, Tensor *res, int dim) override;
        void relu(Tensor *lhs, Tensor *res) override;
        void reluPrime(Tensor *lhs, Tensor *res) override;
        void crossEntropy(Tensor *lhs, const Tensor *labels, Tensor *maxs, Tensor *sums, Tensor *res) override;
        void crossEntropyBackward(Tensor *lhs, const Tensor *labels, Tensor *maxs, Tensor *sums, Tensor *res) override;
        void calcAllGradNorm(const std::vector<Tensor*> &grads, Tensor *norm) override;
        void clipGrad(Tensor *grad, const Tensor *norm, float grad_clip_val) override;

        // Memory management
        void* alloc(size_t size) override;
        void memset(void* ptr, int value, size_t size) override;
        void memcpy(void* dst, const void* src, size_t size) override;
        void free(void* ptr) override;
        float get_float(const Tensor *tensor, int index) override;
};

#endif