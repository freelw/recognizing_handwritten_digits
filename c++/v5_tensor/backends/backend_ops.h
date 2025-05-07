#ifndef BACKEND_OPS_H
#define BACKEND_OPS_H

#include "tensor/tensor.h"

class BackendOps {
    public:
        BackendOps() = default;
        virtual ~BackendOps() = default;
        virtual void add(Tensor *lhs, const Tensor *rhs, Tensor *res) = 0;
        virtual void addEq(Tensor *lhs, const Tensor *rhs) = 0;
        virtual void expandAdd(Tensor *lhs, const Tensor *rhs, Tensor *res) = 0;
        virtual void at(Tensor *lhs, const Tensor *rhs, Tensor *res) = 0;
        virtual void mul(Tensor *lhs, const Tensor *rhs, Tensor *res) = 0;
        virtual void sum(Tensor *lhs, Tensor *res, int dim) = 0;
        virtual void relu(Tensor *lhs, Tensor *res) = 0;
        virtual void reluPrime(Tensor *lhs, Tensor *res) = 0;
        virtual void crossEntropy(Tensor *lhs, const Tensor *labels, Tensor *maxs, Tensor *sums, Tensor *res) = 0;
        virtual void crossEntropyBackward(Tensor *lhs, const Tensor *labels, Tensor *maxs, Tensor *sums, Tensor *res) = 0;
        virtual void calcAllGradNorm(const std::vector<Tensor*> &grads, Tensor *norm) = 0;
        virtual void clipGrad(Tensor *grad, const Tensor *norm, float grad_clip_val) = 0;

        // Memory management
        virtual void *alloc(size_t size) = 0;
        virtual void memset(void *ptr, int value, size_t size) = 0;
        virtual void memcpy(void *dst, const void *src, size_t size) = 0;
        virtual void free(void *ptr) = 0;
        virtual float get_float(const Tensor *tensor, int index) = 0;
};

extern BackendOps *g_backend_ops;
#endif