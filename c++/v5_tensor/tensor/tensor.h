#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <cassert>
#include <ostream>
#include <string>

#define TENSOR_PADDING_SIZE 16

enum TensorDType {
    INT8 = 0,
    INT16 = 1,
    INT32,
    INT64,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    BOOL
};

std::string TensorDtype_to_string(TensorDType dtype);

class Tensor {
    public:
        Tensor(const std::vector<int> &_shape, const std::string &_name, TensorDType _dtype);
        Tensor(const std::vector<int> &_shape, TensorDType _dtype);
        Tensor(const std::vector<int> &_shape, const std::vector<int> &_strides, const std::string &_name, TensorDType _dtype);
        virtual ~Tensor() = default;
        virtual void set_data(void *ptr);
        virtual void *get_data() const { return data; }
        virtual int size() const;
        virtual int length() const;
        virtual int capacity() const;
        virtual bool sanitize() const;
        virtual bool is_view() const { return false; }
        std::vector<int> get_shape() const { return shape; }
        std::vector<int> get_strides() const { return strides; }
        virtual int get_rank() const { return shape.size(); }
        TensorDType get_dtype() const { return dtype; }
        virtual std::string get_name() const { return name; }
        float *location(const std::vector<int> &indices) const;
        Tensor *transpose();
        Tensor *fill(float value);
        friend std::ostream &operator<<(std::ostream &output, const Tensor &s);
    protected:
        int cell_size() const;
    protected:
        std::vector<int> shape;
        std::vector<int> strides;
        std::string name;
        TensorDType dtype;
    private:
        void *data;
};

class TensorView : public Tensor {
    public:
        TensorView(Tensor *_parent, const std::vector<int> &shape, const std::vector<int> &strides, const std::string &name)
            : Tensor(shape, strides, name, _parent->get_dtype()), parent(_parent) {}
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
        int length() const override {
            return parent->length();
        }
        int capacity() const override {
            return parent->capacity();
        }
        bool sanitize() const override {
            return parent->sanitize();
        }
        virtual std::string get_name() const { return name + "_view"; }
    private:
        Tensor *parent;
};

extern std::vector<Tensor*> g_tensors;
extern std::vector<Tensor*> g_tensor_views;
extern std::vector<Tensor*> g_grad_tensors;

Tensor *allocTensor(const std::vector<int> &shape, const std::string &name, TensorDType _dtype = FLOAT32);
Tensor *allocTensor(const std::vector<int> &shape, TensorDType _dtype = FLOAT32);
Tensor *allocTensorView(Tensor *parent, const std::vector<int> &shape, const std::vector<int> &strides, const std::string &name);
Tensor *allocGradTensor(const std::vector<int> &shape, const std::string &name);
Tensor *allocGradTensor(const std::vector<int> &shape);
void printAllTensors();

void freeAllTensors();
void freeAllTensorViews();
void freeAllGradTensors();

extern void *grad_tensors_data;
extern size_t grad_tensors_data_capacity;
void allocMemAndInitTensors();
void releaseTensorMem();
void sanitizeTensors();

#endif