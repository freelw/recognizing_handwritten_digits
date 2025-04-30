#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <cassert>
#include <ostream>
#include <string>

#define TENSOR_PADDING_SIZE 16

class Tensor {
    public:
        Tensor(std::vector<int> _shape, const std::string &_name);
        Tensor(std::vector<int> _shape);
        ~Tensor();
        virtual void set_data(void *ptr);
        virtual void *get_data() const { return data; }
        virtual int size() const;
        virtual int capacity() const;
        virtual bool sanitize() const;
        virtual bool is_view() const { return false; }
        std::vector<int> get_shape() const { return shape; }
        virtual int get_rank() const { return shape.size(); }
        virtual std::string get_name() const { return name; }
        friend std::ostream &operator<<(std::ostream &output, const Tensor &s) {
            output << "Tensor(" << s.get_name() << ")(";
            for (size_t i = 0; i < s.shape.size(); ++i) {
                output << s.shape[i];
                if (i != s.shape.size() - 1) {
                    output << ", ";
                }
            }
            output << ")";
            return output;
        }
    protected:
        std::vector<int> shape;
        std::vector<int> strides;
        std::string name;
    private:
        void *data;
};

class TensorView : public Tensor {
    public:
        TensorView(Tensor *parent, const std::vector<int> &shape, const std::string &name)
            : Tensor(shape, name), parent(parent) {}
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
        int get_rank() const override {
            return parent->get_rank();
        }
        virtual std::string get_name() const { return name + "_view"; }
    private:
        Tensor *parent;
};

extern std::vector<Tensor*> g_tensors;
extern std::vector<Tensor*> g_tensor_views;
extern std::vector<Tensor*> g_grad_tensors;

Tensor *allocTensor(const std::vector<int> &shape, const std::string &name);
Tensor *allocTensor(const std::vector<int> &shape);
Tensor *allocTensorView(Tensor *parent, const std::vector<int> &shape, const std::string &name);
Tensor *allocGradTensor(const std::vector<int> &shape, const std::string &name);
Tensor *allocGradTensor(const std::vector<int> &shape);
void printAllTensors();

void freeAllTensors();
void freeAllTensorViews();
void freeAllGradTensors();

#endif