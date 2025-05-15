#ifndef ACTIONS_H
#define ACTIONS_H

#include "tensor/tensor.h"
#include <ostream>
#include <string>

class Parameter;

class Action {
    public:
        Action(Tensor *_lhs, const Tensor *_rhs, Tensor *_res)
            : lhs(_lhs), rhs(_rhs), res(_res), exec_times(0) {}
        virtual ~Action() = default;
        virtual void execute() = 0;
        virtual std::string to_string() const {
            return "Action not implemented";
        }
        virtual bool is_do_once() const;
        virtual bool is_backward_boundary();
        bool executed_once() const;
        void increase_exec_times();
        int get_exec_times() const;
        friend std::ostream &operator<<(std::ostream &output, const Action &);
    protected:
        Tensor *lhs;
        const Tensor *rhs;
        Tensor *res;
        int exec_times;
};

class AddAction : public Action {
    public:
        AddAction(Tensor *_lhs, const Tensor *_rhs, Tensor *_res)
            : Action(_lhs, _rhs, _res) {}
        void execute() override;
        std::string to_string() const override;
};

class AddEqAction : public Action {
    public:
        AddEqAction(Tensor *_lhs, const Tensor *_rhs)
            : Action(_lhs, _rhs, nullptr) {}
        void execute() override;
        std::string to_string() const override;
};

class ExpandAddAction : public Action {
    public:
        ExpandAddAction(Tensor *_lhs, const Tensor *_rhs, Tensor *_res)
            : Action(_lhs, _rhs, _res) {}
        void execute() override;
        std::string to_string() const override;
};

class AtAction : public Action {
    public:
        AtAction(Tensor *_lhs, const Tensor *_rhs, Tensor *_res)
            : Action(_lhs, _rhs, _res) {}
        void execute() override;
        std::string to_string() const override;
};

class MulAction : public Action {
    public:
        MulAction(Tensor *_lhs, const Tensor *_rhs, Tensor *_res)
            : Action(_lhs, _rhs, _res) {}
        void execute() override;
        std::string to_string() const override;
};

class SumAction : public Action {
    public:
        SumAction(Tensor *_lhs, Tensor *_res, int _dim)
            : Action(_lhs, nullptr, _res), dim(_dim) {}
        void execute() override;
        std::string to_string() const override;
    private:
        int dim;
};

class ReluAction : public Action {
    public:
        ReluAction(Tensor *_lhs, Tensor *_res)
            : Action(_lhs, nullptr, _res) {}
        void execute() override;
        std::string to_string() const override;
};

class ReluPrimeAction : public Action {
    public:
        ReluPrimeAction(Tensor *_lhs, Tensor *_res)
            : Action(_lhs, nullptr, _res) {}
        void execute() override;
        std::string to_string() const override;
};

class CrossEntropyAction : public Action {
    public:
        CrossEntropyAction(Tensor *_lhs, const Tensor *labels, Tensor *_maxs, Tensor *_sums, Tensor *_res)
            : Action(_lhs, labels, _res), maxs(_maxs), sums(_sums) {}
        void execute() override;
        std::string to_string() const override;
    private:
        Tensor *maxs;
        Tensor *sums;
};

class CrossEntropyBackwardAction : public Action {
    public:
        CrossEntropyBackwardAction(Tensor *_lhs, const Tensor *labels, Tensor *_maxs, Tensor *_sums, Tensor *_res)
            : Action(_lhs, labels, _res), maxs(_maxs), sums(_sums) {}
        void execute() override;
        std::string to_string() const override;
    private:
        Tensor *maxs;
        Tensor *sums;
};

class CalcAllGradNormAction : public Action {
    public:
        CalcAllGradNormAction(const std::vector<Tensor*> &_grads, Tensor *_norm)
            : Action(nullptr, nullptr, _norm), grads(_grads) {}
        void execute() override;
        std::string to_string() const override;
    private:
        std::vector<Tensor*> grads;
};

class ClipGradAction : public Action {
    public:
        ClipGradAction(Tensor *_grad, Tensor *_norm, float _grad_clip_val)
            : Action(_grad, _norm, nullptr), grad_clip_val(_grad_clip_val) {}
        void execute() override;
        std::string to_string() const override;
    private:
        float grad_clip_val;
};

class AdamStepAction : public Action {
    public:
        AdamStepAction(Parameter *_param, float _lr, float _beta1, float _beta2, float _epsilon)
            : Action(nullptr, nullptr, nullptr), param(_param), lr(_lr), beta1(_beta1), beta2(_beta2), epsilon(_epsilon) {}
        void execute() override;
        std::string to_string() const override;
    private:
        Parameter *param;
        float lr;
        float beta1;
        float beta2;
        float epsilon;
};

class ZeroGradAction : public Action {
    public:
        ZeroGradAction()
            : Action(nullptr, nullptr, nullptr) {}
        void execute() override;
        std::string to_string() const override;
};

class InitWeightAction : public Action {
    public:
        InitWeightAction(Tensor *_lhs, const std::string &_init_type, float _sigma, float _mean)
            : Action(_lhs, nullptr, nullptr), init_type(_init_type), sigma(_sigma), mean(_mean) {}
        void execute() override;
        bool is_do_once() const override {
            return true;
        }
        std::string to_string() const override;
    private:
        std::string init_type;
        float sigma;
        float mean;
};

class BoundaryAction : public Action {
    public:
        BoundaryAction()
            : Action(nullptr, nullptr, nullptr) {}
        void execute() override;
        bool is_backward_boundary() override;
        std::string to_string() const override;
};

class AssignShapeAndStridesAction : public Action {
    public:
        AssignShapeAndStridesAction(
            Tensor *tensor_shape,
            Tensor *tensor_strides,
            const std::vector<int> &_shape,
            const std::vector<int> &_strides
        );
        virtual ~AssignShapeAndStridesAction();
        void execute() override;
        std::string to_string() const override;
    private:
        std::vector<int> shape;
        std::vector<int> strides;
        int32_t *shape_data;
        int32_t *strides_data;
};

class ReshapeDeepCpAction : public Action {
    public:
        ReshapeDeepCpAction(
            Tensor *_lhs, const Tensor *_rhs,
            const Tensor *_shape, const Tensor *_strides)
            : Action(_lhs, _rhs, nullptr),
            shape(_shape), strides(_strides) {}
        void execute() override;
        std::string to_string() const override;
    private:
        const Tensor *shape;
        const Tensor *strides;
};

class RepeatInterleaveAction : public Action {
    public:
        RepeatInterleaveAction(Tensor *_lhs, Tensor *_res, int _n)
            : Action(_lhs, nullptr, _res), n(_n) {}
        void execute() override;
        std::string to_string() const override;
    private:
        int n;
};

class SequenceMaskAction : public Action {
    public:
        SequenceMaskAction(Tensor *_lhs, const Tensor *_rhs, Tensor *_res, float _value)
            : Action(_lhs, _rhs, _res), value(_value) {}
        void execute() override;
        std::string to_string() const override;
    private:
        float value;
};

class SoftmaxAction : public Action {
    public:
        SoftmaxAction(Tensor *_lhs, Tensor *_res)
            : Action(_lhs, nullptr, _res) {}
        void execute() override;
        std::string to_string() const override;
};

std::vector<Action *> getOnceActions();
void gCreateAction(Action *action);
void gDoActions();
void gDoForwardActions();
void printAllActions();
void freeAllActions();

#endif