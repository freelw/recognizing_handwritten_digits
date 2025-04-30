#include "tensor.h"
#include "backends/backend_ops.h"
#include "graph/actions.h"

Tensor::Tensor(std::vector<int> _shape, const std::string &_name)
    : shape(_shape), data(nullptr), name(_name) {
    strides.resize(shape.size());
    strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

Tensor::Tensor(std::vector<int> _shape)
    : Tensor(_shape, "") {
    
}

Tensor::~Tensor() {
    
}

void Tensor::set_data(void *ptr) {
    data = ptr;
}

int Tensor::size() const {
    int total_size = 1;
    for (int dim : shape) {
        total_size *= dim;
    }
    return total_size;
}

int Tensor::capacity() const {
    return size() + TENSOR_PADDING_SIZE;
}

bool Tensor::sanitize() const {
    if (data == nullptr) {
        return false;
    }
    char * data_ptr = reinterpret_cast<char*>(data)+size();
    for (int i = 0; i < TENSOR_PADDING_SIZE; ++i) {
        if (data_ptr[i] != 0) {
            return false;
        }
    }
    return true;
}

std::vector<Tensor*> g_tensors;
std::vector<Tensor*> g_tensor_views;
std::vector<Tensor*> g_grad_tensors;

Tensor *allocTensor(const std::vector<int> &shape, const std::string &name) {
    Tensor *tensor = new Tensor(shape, name);
    g_tensors.push_back(tensor);
    return tensor;
}

Tensor *allocTensor(const std::vector<int> &shape) {
    return allocTensor(shape, "tensor_autoname");
}

Tensor *allocTensorView(Tensor *parent, const std::vector<int> &shape, const std::string &name) {
    Tensor *tensor_view = new TensorView(parent, shape, name);
    g_tensor_views.push_back(tensor_view);
    return tensor_view;
}

Tensor *allocGradTensor(const std::vector<int> &shape, const std::string &name) {
    Tensor *grad_tensor = new Tensor(shape, name);
    g_grad_tensors.push_back(grad_tensor);
    return grad_tensor;
}

Tensor *allocGradTensor(const std::vector<int> &shape) {
    return allocGradTensor(shape, "grad_autoname");
}

void printAllTensors() {
    std::cout << "Tensors:" << std::endl;
    for (Tensor *tensor : g_tensors) {
        std::cout << "\t" << *tensor << std::endl;
    }
    std::cout << "Tensor Views:" << std::endl;
    for (Tensor *tensor_view : g_tensor_views) {
        std::cout << "\t" << *tensor_view << std::endl;
    }
    std::cout << "Grad Tensors:" << std::endl;
    for (Tensor *grad_tensor : g_grad_tensors) {
        std::cout << "\t" << *grad_tensor << std::endl;
    }
}

void freeAllTensors() {
    for (Tensor *tensor : g_tensors) {
        delete tensor;
    }
    g_tensors.clear();
}

void freeAllTensorViews() {
    for (Tensor *tensor_view : g_tensor_views) {
        delete tensor_view;
    }
    g_tensor_views.clear();
}

void freeAllGradTensors() {
    for (Tensor *grad_tensor : g_grad_tensors) {
        delete grad_tensor;
    }
    g_grad_tensors.clear();
}