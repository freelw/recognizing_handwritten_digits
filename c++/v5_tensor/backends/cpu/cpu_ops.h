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
        void adamStep(Tensor *w, Tensor *grad, Tensor *m, Tensor *v, int t, float lr, float beta1, float beta2, float epsilon) override;
        void init_weight_gauss(Tensor *tensor, float mean, float sigma) override;
        void init_weight_uniform(Tensor *tensor, float sigma) override;
        void fill(Tensor *tensor, float value) override;

        // Memory management
        void* alloc(size_t size) override;
        void memset(void* ptr, int value, size_t size) override;
        void cp_device_to_device(void* dst, const void* src, size_t size) override;
        void free(void* ptr) override;
        void cp_to_device(Tensor *dst_tensor, char *src, size_t size) override;
        void cp_from_device(char *dst, Tensor *src_tensor, size_t size) override;
};

#endif