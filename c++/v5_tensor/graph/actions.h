#ifndef ACTIONS_H
#define ACTIONS_H

#include "tensor/tensor.h"

class Action {
    public:
        Action(Tensor *_lhs, const Tensor *_rhs, Tensor *_res)
            : lhs(_lhs), rhs(_rhs), res(_res) {}
        virtual void execute() = 0;
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
};

class AddEqAction : public Action {
    public:
        AddEqAction(Tensor *_lhs, const Tensor *_rhs)
            : Action(_lhs, _rhs, nullptr) {}
        void execute() override;
};

class AtAction : public Action {
    public:
        AtAction(Tensor *_lhs, const Tensor *_rhs, Tensor *_res)
            : Action(_lhs, _rhs, _res) {}
        void execute() override;
};

class MulAction : public Action {
    public:
        MulAction(Tensor *_lhs, const Tensor *_rhs, Tensor *_res)
            : Action(_lhs, _rhs, _res) {}
        void execute() override;
};

class SumAction : public Action {
    public:
        SumAction(Tensor *_lhs, Tensor *_res, int _dim)
            : Action(_lhs, nullptr, _res), dim(_dim) {}
        void execute() override;
    private:
        int dim;
};

void gCreateAction(Action *action);
void freeAllActions();

#endif