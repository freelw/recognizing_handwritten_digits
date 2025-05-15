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
        virtual void emb_at(Tensor *lhs, const Tensor *indices, const Tensor *rhs, Tensor *res) = 0;
        virtual void mul(Tensor *lhs, const Tensor *rhs, Tensor *res) = 0;
        virtual void sum(Tensor *lhs, Tensor *res, int dim) = 0;
        virtual void relu(Tensor *lhs, Tensor *res) = 0;
        virtual void reluPrime(Tensor *lhs, Tensor *res) = 0;
        virtual void crossEntropy(Tensor *lhs, const Tensor *labels, Tensor *maxs, Tensor *sums, Tensor *res) = 0;
        virtual void crossEntropyBackward(Tensor *lhs, const Tensor *labels, Tensor *maxs, Tensor *sums, Tensor *res) = 0;
        virtual void calcAllGradNorm(const std::vector<Tensor*> &grads, Tensor *norm) = 0;
        virtual void clipGrad(Tensor *grad, const Tensor *norm, float grad_clip_val) = 0;
        virtual void adamStep(Tensor *w, Tensor *grad, Tensor *m, Tensor *v, int t, float lr, float beta1, float beta2, float epsilon) = 0;
        virtual void init_weight_gauss(Tensor *tensor, float mean, float sigma) = 0;
        virtual void init_weight_uniform(Tensor *tensor, float sigma) = 0;
        virtual void init_weight_for_dbg(Tensor *tensor, float scale) = 0;
        virtual void fill(Tensor *tensor, float value) = 0;
        virtual void reshape_deep_cp(
            Tensor *dst_tensor, const Tensor *src_tensor,
            const Tensor *src_shape, const Tensor *src_strides
        ) = 0;
        virtual void repeat_interleave(Tensor *lhs, Tensor *res, int n) = 0;
        virtual void sequence_mask(Tensor *lhs, const Tensor *mask, Tensor *res, float value) = 0;
        virtual void softmax(Tensor *lhs, Tensor *res, Tensor *maxs, Tensor *sums) = 0;
        
        // Memory management
        virtual void *alloc(size_t size) = 0;
        virtual void memset(void *ptr, int value, size_t size) = 0;
        virtual void cp_device_to_device(void *dst, const void *src, size_t size) = 0;
        virtual void free(void *ptr) = 0;
        virtual void cp_to_device(Tensor *dst_tensor, char *src, size_t size) = 0;
        virtual void cp_from_device(char *dst, const Tensor *src_tensor, size_t size) = 0;
};

extern BackendOps *g_backend_ops;
#endif