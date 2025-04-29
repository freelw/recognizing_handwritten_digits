#include "tensor.h"

Tensor::Tensor(std::vector<int> _shape) : shape(_shape), data(nullptr) {
    strides.resize(shape.size());
    strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
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

Tensor *Tensor::transpose_2d() {
    if (shape.size() != 2) {
        std::cerr << "Error: Transpose only supported for 2D tensors" << std::endl;
        return nullptr;
    }
    if (data != nullptr) {
        std::cerr << "Error: Transpose not supported for non-null data" << std::endl;
        abort();
    }
    
    Tensor *transposed_tensor = allocTensorView(this);
    transposed_tensor->shape[0] = shape[1];
    transposed_tensor->shape[1] = shape[0];
    transposed_tensor->strides[0] = strides[1];
    transposed_tensor->strides[1] = strides[0];
    return transposed_tensor;
}

std::vector<Tensor*> g_tensors;
std::vector<Tensor*> g_tensor_views;

Tensor *allocTensor(const std::vector<int> &shape) {
    Tensor *tensor = new Tensor(shape);
    g_tensors.push_back(tensor);
    return tensor;
}

Tensor *allocTensorView(Tensor *parent) {
    Tensor *tensor_view = new TensorView(parent);
    g_tensor_views.push_back(tensor_view);
    return tensor_view;
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