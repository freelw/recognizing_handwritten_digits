#ifndef ACTIONS_H
#define ACTIONS_H

#include "tensor/tensor.h"
#include <ostream>
#include <string>

class Parameter;

class Action {
    public:
        Action(Tensor *_lhs, const Tensor *_rhs, Tensor *_res)
            : lhs(_lhs), rhs(_rhs), res(_res) {}
        virtual void execute() = 0;
        virtual std::string to_string() const {
            return "Action not implemented";
        }
        friend std::ostream &operator<<(std::ostream &output, const Action &);
    protected:
        Tensor *lhs;
        const Tensor *rhs;
        Tensor *res;
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

void gCreateAction(Action *action);
void gDoActions();
void printAllActions();
void freeAllActions();

#endif