#ifndef BACKEND_OPS_H
#define BACKEND_OPS_H

class BackendOps {
    public:
        BackendOps() = default;
        virtual ~BackendOps() = default;
        virtual void add(Tensor *lhs, const Tensor *rhs, Tensor *res) = 0;
        virtual void addEq(Tensor *lhs, const Tensor *rhs) = 0;
        virtual void at(Tensor *lhs, const Tensor *rhs, Tensor *res) = 0;
        virtual void mul(Tensor *lhs, const Tensor *rhs, Tensor *res) = 0;
        virtual void sum(Tensor *lhs, int dim, Tensor *res) = 0;
};

extern BackendOps *g_backend_ops;
#endif