#include "tensor.h"

Tensor::Tensor(std::vector<int> _shape) : data(nullptr), shape(_shape) {
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