#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <cassert>

#define TENSOR_PADDING_SIZE 16

class Tensor;

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

class Tensor {
    public:
        Tensor(std::vector<int> _shape);
        ~Tensor();
        virtual void set_data(void *ptr);
        virtual void set_grad(void *ptr);
        virtual void *get_data() const { return data; }
        virtual void *get_grad() const { return grad; }
        virtual int size() const;
        virtual int capacity() const;
        virtual bool sanitize() const;
        virtual bool is_view() const { return false; }
        std::vector<int> get_shape() const { return shape; }
        Tensor *transpose_2d();
        Tensor *operator+=(const Tensor *other);
        Tensor *at(const Tensor *other);
        Tensor *operator*(const Tensor *other);
        virtual int get_rank() const { return shape.size(); }
    protected:
        std::vector<int> shape;
        std::vector<int> strides;
    private:
        void *data;
        void *grad;
};

class TensorView : public Tensor {
    public:
        TensorView(Tensor *parent)
            : Tensor(parent->get_shape()), parent(parent) {}
        bool is_view() const override { return true; }
        void set_data(void *ptr) override {
            std::cerr << "Error: Cannot set data for TensorView" << std::endl;
            assert(false);
        }
        void set_grad(void *ptr) override {
            std::cerr << "Error: Cannot set grad for TensorView" << std::endl;
            assert(false);
        }
        void *get_data() const override {
            return parent->get_data();
        }
        void *get_grad() const override {
            return parent->get_grad();
        }
        int size() const override {
            return parent->size();
        }
        int capacity() const override {
            return parent->capacity();
        }
        bool sanitize() const override {
            return parent->sanitize();
        }
        int get_rank() const override {
            return parent->get_rank();
        }
    private:
        Tensor *parent;
};

Tensor *allocTensor(const std::vector<int> &shape);
Tensor *allocTensorView(Tensor *parent);

void freeAllTensors();
void freeAllTensorViews();

#endif