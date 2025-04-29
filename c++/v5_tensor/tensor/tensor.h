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
    int size() const;
    int capacity() const;
    bool sanitize() const;
    virtual bool is_view() const { return false; }
private:
    void *data;
    std::vector<int> shape;
    std::vector<int> strides;
};

class TensorView : public Tensor {
public:
    TensorView(std::vector<int> _shape, Tensor *parent)
        : Tensor(_shape), parent(parent) {}
    bool is_view() const override { return true; }
    void set_data(void *ptr) override {
        std::cerr << "Error: Cannot set data for TensorView" << std::endl;
        assert(false);
    }
private:
    Tensor *parent;
};

#endif