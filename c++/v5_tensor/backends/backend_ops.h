#ifndef BACKEND_OPS_H
#define BACKEND_OPS_H

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
        virtual void zero_grad();
        virtual void *alloc(int64_t size) = 0;
        virtual void *memset(void *ptr, int value, int64_t size) = 0;
};

extern BackendOps *g_backend_ops;
#endif