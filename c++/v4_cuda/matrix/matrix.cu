#include "matrix.cuh"

#include <iostream>
#include <assert.h>
#include <string.h>
#include <vector>
#include <omp.h> // Include OpenMP header
#include <random>
#include <chrono>
#include "backends/cpu/cpu_ops.cuh"

Matrix::Matrix(Shape _shape)
    : initialized(false),
    allocated(false),
    shape(_shape),
    commited(false),
    data_device(nullptr) {
    data = new DATATYPE[shape.size()];
    data_device = g_backend_ops->allocDeviceMem(shape.size() * sizeof(DATATYPE));
    allocated = true;
    zero();
    this->cp_to_device();
}

Matrix::Matrix(const Matrix &m):
    initialized(m.initialized),
    allocated(false),
    shape(m.shape),
    commited(false),
    data_device(nullptr) {
    assert(initialized);
    data = new DATATYPE[shape.size()];
    data_device = g_backend_ops->allocDeviceMem(shape.size() * sizeof(DATATYPE));
    allocated = true;
    memcpy(data, m.data, sizeof(DATATYPE) * shape.rowCnt * shape.colCnt);
    this->cp_to_device();
}

Matrix::Matrix(const std::vector<DATATYPE> &v):
    initialized(false),
    allocated(false),
    shape(Shape(v.size(), 1)),
    commited(false),
    data_device(nullptr) {
    data = new DATATYPE[shape.size()];
    allocated = true;
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        data[i] = v[i];
    }
    initialized = true;
    this->cp_to_device();
}

Matrix::~Matrix() {
    assert(initialized && allocated);
    delete [] data;
    data = nullptr;
    g_backend_ops->releaseDeviceMem(data_device);
    data_device = nullptr;
}

Matrix *Matrix::zero() {
    assert(allocated);
    memset(data, 0, sizeof(DATATYPE) * shape.size());
    initialized = true;
    return this;
}

bool Matrix::checkShape(const Matrix &m) {
    if (!(this->getShape() == m.getShape())) {
        std::cerr << 
            "matrix shape missmatch." << 
            this->getShape() << " vs " << m.getShape()<< 
            std::endl;
        assert(false);
    }
    if (!m.initialized) {
        std::cerr << "matrix not initialized..." << std::endl;
        assert(false);
    }
    return true;
}

ostream &operator<<(ostream &output, const Matrix &m) {
    if (!m.initialized) {
        output << "matrix not initialized." << endl;
        return output;
    }
    output << "[";
    for (uint i = 0; i < m.shape.rowCnt; ++ i) {
        if (i > 0) {
            output << " ";
        }
        output << "[";
        for (uint j = 0; j < m.shape.colCnt-1; ++ j) {
            output << m[i][j] << ", ";
        }
        output << m[i][m.shape.colCnt-1] << "]";
        if (i < m.shape.rowCnt-1) {
            output << endl;
        }
    }
    output << "]" << endl;
    return output;
}

Matrix *Matrix::expand_add(const Matrix &m) {
    assert(m.shape.rowCnt == shape.rowCnt);
    assert(m.shape.colCnt == 1);
    Matrix *res = allocTmpMatrix(this);
    g_backend_ops->expand_add(res, m);
    return res;
}

Matrix *Matrix::operator+(const Matrix &m) {
    checkShape(m);
    Matrix *res = allocTmpMatrix(this);
    g_backend_ops->operator_add(res, m);
    return res;
}

Matrix *Matrix::operator+=(const Matrix &m) {
    checkShape(m);
    g_backend_ops->operator_add(this, m);
    return this;
}

Matrix *Matrix::pow2() {
    Matrix *res = allocTmpMatrix(this);
    g_backend_ops->pow2(res);
    return res;
}

Matrix *Matrix::operator+(DATATYPE dt) {
    Matrix *res = allocTmpMatrix(this);
    g_backend_ops->operator_add_val(res, dt);
    return res;
}

Matrix *Matrix::operator-(DATATYPE dt) {
    Matrix *res = allocTmpMatrix(this);
    g_backend_ops->operator_minus_val(res, dt);
    return res;
}

Matrix *Matrix::operator-() {
    Matrix *res = allocTmpMatrix(this);
    g_backend_ops->operator_negative(res);
    return res;
}

Matrix *operator-(DATATYPE v, const Matrix &m) {
    Matrix *res = allocTmpMatrix(m);
    g_backend_ops->operator_val_minus(v, res);
    return res;
}

Matrix *Matrix::operator-(const Matrix &m) {
    checkShape(m);
    Matrix *res = allocTmpMatrix(this);
    g_backend_ops->operator_minus(res, m);
    return res;
}

Matrix *Matrix::operator-=(const Matrix &m) {
    checkShape(m);
    g_backend_ops->operator_minus(this, m);
    return this;
}

Matrix *Matrix::operator*(const Matrix &m) {
    checkShape(m);
    Matrix *res = allocTmpMatrix(this);
    g_backend_ops->operator_multiply(res, m);
    return res;
}

Matrix *Matrix::operator*(DATATYPE v) {
    Matrix *res = allocTmpMatrix(this);
    g_backend_ops->operator_multiply_val(res, v);
    return res;
}

Matrix *Matrix::operator*=(DATATYPE v) {
    g_backend_ops->operator_multiply_val(this, v);
    return this;
}

Matrix *Matrix::operator/(DATATYPE v) {
    Matrix *res = allocTmpMatrix(this);
    g_backend_ops->operator_divide_val(res, v);
    return res;
}

Matrix *Matrix::Relu() {
    Matrix *res = allocTmpMatrix(this);
    g_backend_ops->Relu(res);
    return res;
}

Matrix *Matrix::Relu_prime() {
    Matrix *res = allocTmpMatrix(this);
    g_backend_ops->Relu_prime(res);
    return res;
}

Matrix *Matrix::tanh() {
    Matrix *res = allocTmpMatrix(this);
    g_backend_ops->tanh(res);
    return res;
}

Matrix *Matrix::tanh_prime() {
    Matrix *res = allocTmpMatrix(this);
    g_backend_ops->tanh_prime(res);
    return res;
}

Matrix& Matrix::operator=(const Matrix &m) {
    assert(m.initialized);
    this->reShape(m.shape);
    g_backend_ops->operator_equal(this, m);
    return *this;
}

DATATYPE *Matrix::operator[](unsigned int index) const {
    assert(!g_backend_ops->is_gpu());
    assert(index < shape.rowCnt);
    return (DATATYPE *)&(data[index*shape.colCnt]);
}

Shape Matrix::getShape() const {
    return shape;
}

Matrix *Matrix::at(const Matrix &m) {
    assert(m.shape.rowCnt == shape.colCnt);
    Matrix *res = allocTmpMatrix(Shape(shape.rowCnt, m.shape.colCnt));
    g_backend_ops->operator_at(res, this, m);
    return res;
}

Matrix *Matrix::transpose() {
    Matrix *res = allocTmpMatrix(Shape(shape.colCnt, shape.rowCnt));
    g_backend_ops->operator_transpose(res, this);
    return res;
}

bool Matrix::valid(uint x, uint y) const {
    return allocated && initialized && x < shape.rowCnt && y < shape.colCnt;
}

void Matrix::reShape(Shape _shape) {
    assert(allocated && initialized);
    delete []data;
    g_backend_ops->releaseDeviceMem(data_device);
    shape = _shape;
    data = new DATATYPE[shape.size()];
    zero();
    data_device = g_backend_ops->allocDeviceMem(shape.size() * sizeof(DATATYPE));
    cp_to_device();
}

Matrix *Matrix::assign(Matrix *other) {
    assert(allocated && initialized);
    checkShape(other->getShape());
    memcpy(data, other->data, sizeof(DATATYPE) * shape.size());
    return this;
}

Matrix *Matrix::sum(uint dim) {
    assert(dim == 1);
    if (dim == 1) {
        Matrix *res = allocTmpMatrix(Shape(shape.rowCnt, 1));
        for (uint i = 0; i < shape.rowCnt; ++ i) {
            for (uint j = 0; j < shape.colCnt; ++ j) {
                (*res)[i][0] += (*this)[i][j];
            }
        }
        return res;
    }
    return nullptr;
}

std::vector<Matrix *> Matrix::split(uint dim) {
    assert(dim == 1);
    if (dim == 1) {
        std::vector<Matrix *> res;
        for (uint i = 0; i < shape.colCnt; ++ i) {
            Matrix *m = allocTmpMatrix(Shape(shape.rowCnt, 1));
            for (uint j = 0; j < shape.rowCnt; ++ j) {
                (*m)[j][0] = (*this)[j][i];
            }
            res.push_back(m);
        }
        return res;
    }
    return {};
}

DATATYPE *Matrix::getData() const {
    return data;
}

Matrix *Matrix::fill(DATATYPE value) {
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        for (uint j = 0; j < shape.colCnt; ++ j) {
            (*this)[i][j] = value;
        }
    }
    return this;
}

std::vector<uint> Matrix::argMax() {
    Shape shape = getShape();
    std::vector<uint> res;
    res.reserve(shape.colCnt);
    for (uint i = 0; i < shape.colCnt; ++ i) {
        uint max_index = 0;
        for (uint j = 1; j < shape.rowCnt; ++ j) {
            if ((*this)[j][i] > (*this)[max_index][i]) {
                max_index = j;
            }
        }
        res.push_back(max_index);
    }
    return res;
}

std::vector<DATATYPE> Matrix::avg() {
    Shape shape = getShape();
    std::vector<DATATYPE> res;
    res.reserve(shape.colCnt);
    for (uint i = 0; i < shape.colCnt; ++ i) {
        DATATYPE sum = 0;
        for (uint j = 0; j < shape.rowCnt; ++ j) {
            sum += (*this)[j][i];
        }
        res.push_back(sum/shape.rowCnt);
    }
    return res;
}

std::vector<DATATYPE> Matrix::var() {
    std::vector<DATATYPE> res;
    std::vector<DATATYPE> avg_res = this->avg();
    Shape shape = getShape();
    for (uint i = 0; i < shape.colCnt; ++ i) {
        DATATYPE sum = 0;
        auto avg_r = avg_res[i];
        for (uint j = 0; j < shape.rowCnt; ++ j) {
            sum += std::pow(((*this)[j][i] - avg_r), 2);
        }
        res.push_back(sum/shape.rowCnt);
    }
    return res;
}

DATATYPE _sigmoid(DATATYPE z) {
    return 1./(1.+exp(-z));
}

Matrix *Matrix::sigmoid() {
    Shape shape = getShape();
    Matrix *res = allocTmpMatrix(this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*res)[i][j] = _sigmoid((*res)[i][j]);
        }
    }
    return res;
}

Matrix *Matrix::sigmoid_prime() {
    return *sigmoid() * *(1 - *sigmoid());
}

std::vector<Matrix *> tmpMatrics;
Matrix *allocTmpMatrix(Matrix *m) {
    return allocTmpMatrix(*m);
}

Matrix *allocTmpMatrix(const Matrix &m) {
    Matrix *res = new Matrix(m);
    tmpMatrics.push_back(res);
    return res;
}

Matrix *allocTmpMatrix(const Shape & shape) {
    Matrix *res = new Matrix(shape);
    res->zero();
    tmpMatrics.push_back(res);
    return res;
}

Matrix *allocTmpMatrix(const std::vector<DATATYPE> &v) {
    Matrix *res = new Matrix(v);
    tmpMatrics.push_back(res);
    return res;
}

void freeTmpMatrix() {
    for (auto p : tmpMatrics) {
        delete p;
    }
    tmpMatrics.clear();
}

void Matrix::init_weight(DATATYPE sigma, DATATYPE mean) {
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w(seed1);
    std::normal_distribution<DATATYPE> distribution_w(0.0, sigma);
    auto shape = getShape();
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        for (uint j = 0; j < shape.colCnt; ++ j) {
            (*this)[i][j] = distribution_w(generator_w) + mean;
        }
    }
}

void Matrix::init_weight_uniform(DATATYPE sigma) {
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w(seed1);
    std::uniform_real_distribution<DATATYPE> distribution_w(-sigma, sigma);
    auto shape = getShape();
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        for (uint j = 0; j < shape.colCnt; ++ j) {
            (*this)[i][j] = distribution_w(generator_w);
        }
    }
}

void Matrix::set_val(int i, int j, DATATYPE val) {
    assert(i < shape.rowCnt && j < shape.colCnt);
    (*this)[i][j] = val;
    commited = false;
}

DATATYPE Matrix::get_val(int i, int j) const {
    assert(i < shape.rowCnt && j < shape.colCnt);
    return (*this)[i][j];
}

void Matrix::cp_to_device() {
    assert(allocated && initialized);
    commited = true;
    g_backend_ops->cp_to_device(data_device, data, shape.size());
}

void Matrix::cp_from_device() {
    g_backend_ops->cp_from_device(data, data_device, shape.size());
}

TrainingData::TrainingData(int input_layer_size, int _y)
    : y(_y) {  
    x.reserve(input_layer_size);
}

TrainingData::~TrainingData() {
}