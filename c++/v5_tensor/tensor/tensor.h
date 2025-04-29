#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <cassert>

#define TENSOR_PADDING_SIZE 16

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
        // Tensor *transpose_2d();
        // Tensor *expand_add(const Tensor *other);
        // Tensor *operator+=(const Tensor *other);
        // Tensor *at(const Tensor *other);
        // Tensor *operator*(const Tensor *other);
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