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
        virtual void *get_data() const { return data; }
        virtual int size() const;
        virtual int capacity() const;
        virtual bool sanitize() const;
        virtual bool is_view() const { return false; }
        std::vector<int> get_shape() const { return shape; }
        Tensor *transpose_2d();
    protected:
        std::vector<int> shape;
        std::vector<int> strides;
    private:
        void *data;
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
        void *get_data() const override {
            return parent->get_data();
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
    private:
        Tensor *parent;
};

Tensor *allocTensor(const std::vector<int> &shape);
Tensor *allocTensorView(Tensor *parent);

void freeAllTensors();
void freeAllTensorViews();

#endif