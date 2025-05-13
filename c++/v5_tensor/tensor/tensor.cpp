#include "tensor.h"
#include "backends/backend_ops.h"
#include "graph/actions.h"
#include <sstream>

std::string TensorDtype_to_string(TensorDType dtype) {
    switch (dtype) {
        case INT8: return "INT8";
        case INT16: return "INT16";
        case INT32: return "INT32";
        case INT64: return "INT64";
        case FLOAT16: return "FLOAT16";
        case FLOAT32: return "FLOAT32";
        case FLOAT64: return "FLOAT64";
        case BOOL: return "BOOL";
        default: return "UNKNOWN";
    }
}

Tensor::Tensor(const std::vector<int> &_shape, const std::string &_name, TensorDType _dtype)
    : shape(_shape), name(_name), dtype(_dtype), own_storage(true) {
    strides.resize(shape.size());
    strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    storage = new TensorStorage();
}

Tensor::Tensor(const std::vector<int> &_shape, TensorDType _dtype)
    : Tensor(_shape, "", _dtype) {
    
}

Tensor::Tensor(
    const std::vector<int> &_shape,
    const std::vector<int> &_strides,
    const std::string &_name,
    TensorDType _dtype,
    TensorStorage *_storage)
    : shape(_shape), strides(_strides),
    name(_name), dtype(_dtype),
    own_storage(false), storage(_storage) {
    assert(shape.size() == strides.size());
}

Tensor::~Tensor() {
    if (own_storage) {
        delete storage;
    }
}

void Tensor::set_data(void *ptr) {
    storage->data = ptr;
}

int Tensor::size() const {
    return length() * cell_size();
}

int Tensor::length() const {
    int total_length = 1;
    for (int dim : shape) {
        total_length *= dim;
    }
    return total_length;
}

int Tensor::cell_size() const {
    switch (dtype) {
        case INT8: return 1;
        case INT16: return 2;
        case INT32: return 4;
        case INT64: return 8;
        case FLOAT16: return 2;
        case FLOAT32: return 4;
        case FLOAT64: return 8;
        case BOOL: return sizeof(bool);
        default: assert(false); return 0;
    }
}

int Tensor::capacity() const {
    return size() + TENSOR_PADDING_SIZE;
}

Tensor *Tensor::transpose() {
    Tensor *transpose_view = allocTensorView(
        this,
        {this->get_shape()[1], this->get_shape()[0]},
        {this->get_strides()[1], this->get_strides()[0]},
        this->get_name() + "_transpose"
    );
    return transpose_view;
}

Tensor *Tensor::fill(float value) {
    assert(!is_view());
    assert(dtype == FLOAT32);
    g_backend_ops->fill(this, value);
    return this;
}

std::string Tensor::get_meta_info() const {
    std::ostringstream output;
    output << "Tensor";
    if (get_dtype() != FLOAT32) {
        std::string dtype_str = TensorDtype_to_string(get_dtype());
        output << "(" << dtype_str << ")";
    }
    output << "(" << get_name() << ")(";
    for (size_t i = 0; i < shape.size(); ++i) {
        output << shape[i];
        if (i != shape.size() - 1) {
            output << ", ";
        }
    }
    output << ")";
    return output.str();
}

void dfs_print(std::ostream &output, const Tensor &s, void *data, int depth, bool is_start = true) {
    if (!is_start) {
        for (int i = 0; i < depth; ++i) {
            output << " ";
        }
    }
    output << "[";
    auto rank = s.get_rank();
    if (depth == rank-1) {
        auto stride = s.get_strides()[depth];
        auto dtype = s.get_dtype();
        auto length = s.get_shape()[rank-1];
        for (int i = 0; i < length; ++i) {
            if (dtype == FLOAT32) {
                output << *(reinterpret_cast<float*>(data) + i * stride);
            } else if (dtype == INT32) {
                output << *(reinterpret_cast<int32_t*>(data) + i * stride);
            }
            if (i < length - 1) {
                output << ", ";
            } else {
                output << "]";
            }
        }
        return ;
    }
    for (int i = 0; i < s.get_shape()[depth]; ++i) {
        dfs_print(output, s, data, depth+1, i == 0);
        if (i < s.get_shape()[depth] - 1) {
            output << ",";
            for (int j = 0; j < rank - depth -1 ; ++j) {
                output << std::endl;
            }
        }
    }
    output << "]";
}

std::ostream &operator<<(std::ostream &output, const Tensor &s) {
    void *data = ::malloc(s.size());
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(data),
        &s,
        s.size()
    );
    dfs_print(output, s, data, 0);
    ::free(data);
    return output;
}

std::vector<Tensor*> g_tensors;
std::vector<Tensor*> g_tensor_views;
std::vector<Tensor*> g_grad_tensors;

Tensor *allocTensor(const std::vector<int> &shape, const std::string &name, TensorDType dtype) {
    Tensor *tensor = new Tensor(shape, name, dtype);
    g_tensors.push_back(tensor);
    return tensor;
}

Tensor *allocTensor(const std::vector<int> &shape, TensorDType dtype) {
    return allocTensor(shape, "tensor_autoname", dtype);
}

Tensor *allocTensorView(
    Tensor *parent, const std::vector<int> &shape,
    const std::vector<int> &strides, const std::string &name
) {
    Tensor *tensor_view = new Tensor(
        shape, strides, name,
        parent->get_dtype(), parent->get_storage()
    );
    g_tensor_views.push_back(tensor_view);
    return tensor_view;
}

Tensor *allocGradTensor(const std::vector<int> &shape, const std::string &name) {
    Tensor *grad_tensor = new Tensor(shape, name, FLOAT32);
    g_grad_tensors.push_back(grad_tensor);
    return grad_tensor;
}

Tensor *allocGradTensor(const std::vector<int> &shape) {
    return allocGradTensor(shape, "grad_autoname");
}

void printAllTensors() {
    std::cout << "Tensors:" << std::endl;
    for (Tensor *tensor : g_tensors) {
        std::cout << "\t" << tensor->get_meta_info() << std::endl;
    }
    std::cout << "Tensor Views:" << std::endl;
    for (Tensor *tensor_view : g_tensor_views) {
        std::cout << "\t" << tensor_view->get_meta_info() << std::endl;
    }
    std::cout << "Grad Tensors:" << std::endl;
    for (Tensor *grad_tensor : g_grad_tensors) {
        std::cout << "\t" << grad_tensor->get_meta_info() << std::endl;
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

void *tensors_data = nullptr;
void *grad_tensors_data = nullptr;
size_t tensors_data_capacity = 0;
size_t grad_tensors_data_capacity = 0;

void allocMemAndInitTensors() {
    assert(tensors_data == nullptr);
    assert(grad_tensors_data == nullptr);
    assert(tensors_data_capacity == 0);
    assert(grad_tensors_data_capacity == 0);
    
    for (Tensor *tensor : g_tensors) {
        tensors_data_capacity += tensor->capacity();
    }
    for (Tensor *tensor : g_grad_tensors) {
        grad_tensors_data_capacity += tensor->capacity();
    }
    tensors_data = g_backend_ops->alloc(tensors_data_capacity);
    grad_tensors_data = g_backend_ops->alloc(grad_tensors_data_capacity);

    g_backend_ops->memset(tensors_data, 0, tensors_data_capacity);
    g_backend_ops->memset(grad_tensors_data, 0, grad_tensors_data_capacity);

    int64_t offset = 0;
    for (Tensor *tensor : g_tensors) {
        tensor->set_data(reinterpret_cast<char*>(tensors_data) + offset);
        offset += tensor->capacity();
    }

    offset = 0;
    for (Tensor *tensor : g_grad_tensors) {
        tensor->set_data(reinterpret_cast<char*>(grad_tensors_data) + offset);
        offset += tensor->capacity();
    }
}

void releaseTensorMem() {
    assert(tensors_data != nullptr);
    if (grad_tensors_data != nullptr) {
        g_backend_ops->free(grad_tensors_data);
        grad_tensors_data = nullptr;
    }
    g_backend_ops->free(tensors_data);
    tensors_data = nullptr;
    tensors_data_capacity = 0;
    grad_tensors_data_capacity = 0;
}