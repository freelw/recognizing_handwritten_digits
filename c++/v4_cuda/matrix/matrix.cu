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
    data_device(nullptr),
    cpu_ver(0),
    gpu_ver(0) {
    data = new DATATYPE[shape.size()];
    data_device = g_gpu_backend_ops->allocDeviceMem(shape.size() * sizeof(DATATYPE));
    allocated = true;
    zero();
    increase_cpu_ver();
    sync();
}

Matrix::Matrix(const Matrix &m):
    initialized(m.initialized),
    allocated(false),
    shape(m.shape),
    commited(false),
    data_device(nullptr),
    cpu_ver(m.cpu_ver),
    gpu_ver(m.gpu_ver) {
    assert(initialized);
    // assert(m.is_sync());
    data = new DATATYPE[shape.size()];
    data_device = g_gpu_backend_ops->allocDeviceMem(shape.size() * sizeof(DATATYPE));
    g_gpu_backend_ops->deviceMemcpy(data_device, m.data_device, shape.size() * sizeof(DATATYPE));
    allocated = true;
    memcpy(data, m.data, sizeof(DATATYPE) * shape.rowCnt * shape.colCnt);
    increase_cpu_ver();
    sync();
}

Matrix::Matrix(const std::vector<DATATYPE> &v):
    initialized(false),
    allocated(false),
    shape(Shape(v.size(), 1)),
    commited(false),
    data_device(nullptr),
    cpu_ver(0),
    gpu_ver(0) {
    data = new DATATYPE[shape.size()];
    allocated = true;
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        data[i] = v[i];
    }
    data_device = g_gpu_backend_ops->allocDeviceMem(shape.size() * sizeof(DATATYPE));
    initialized = true;
    increase_cpu_ver();
    sync();
}

Matrix::~Matrix() {
    assert(initialized && allocated);
    delete [] data;
    data = nullptr;
    g_gpu_backend_ops->releaseDeviceMem(data_device);
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
    g_backend_ops->operator_relu(res);
    return res;
}

Matrix *Matrix::Relu_prime() {
    Matrix *res = allocTmpMatrix(this);
    g_backend_ops->operator_relu_prime(res);
    return res;
}

Matrix *Matrix::tanh() {
    Matrix *res = allocTmpMatrix(this);
    g_backend_ops->operator_tanh(res);
    return res;
}

Matrix *Matrix::tanh_prime() {
    Matrix *res = allocTmpMatrix(this);
    g_backend_ops->operator_tanh_prime(res);
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

Matrix *Matrix::at(Matrix &m) {
    assert(m.shape.rowCnt == shape.colCnt);
    Matrix *res = allocTmpMatrix(Shape(shape.rowCnt, m.shape.colCnt));
    g_gpu_backend_ops->operator_at(res, this, m);
    // g_backend_ops->operator_at(res, this, m);
    res->sync();
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
    g_gpu_backend_ops->releaseDeviceMem(data_device);
    shape = _shape;
    data = new DATATYPE[shape.size()];
    zero();
    data_device = g_gpu_backend_ops->allocDeviceMem(shape.size() * sizeof(DATATYPE));
    cp_to_device();
}

Matrix *Matrix::assign(Matrix *other) {
    assert(allocated && initialized);
    checkShape(other->getShape());
    g_backend_ops->operator_assign(this, other);
    return this;
}

Matrix *Matrix::sum(uint dim) {
    assert(dim == 1);
    if (dim == 1) {
        Matrix *res = allocTmpMatrix(Shape(shape.rowCnt, 1));
        g_backend_ops->operator_sum(res, this);
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
            res.push_back(m);
        }
        g_backend_ops->operator_split(res, this);
        return res;
    }
    return {};
}

DATATYPE *Matrix::getLowLevelData() const {
    assert(!g_backend_ops->is_gpu());
    return data;
}

DATATYPE *Matrix::getLowLevelDataDevice() const {
    return data_device;
}

Matrix *Matrix::fill(DATATYPE value) {
    g_backend_ops->operator_fill(this, value);
    return this;
}

std::vector<uint> Matrix::argMax() {
    Shape shape = getShape();
    std::vector<uint> res;
    res.reserve(shape.colCnt);
    g_backend_ops->operator_argMax(res, this);
    return res;
}

std::vector<DATATYPE> Matrix::avg() {
    Shape shape = getShape();
    std::vector<DATATYPE> res;
    res.reserve(shape.colCnt);
    g_backend_ops->operator_avg(res, this);
    return res;
}

std::vector<DATATYPE> Matrix::var() {
    Shape shape = getShape();
    std::vector<DATATYPE> res;
    res.reserve(shape.colCnt);
    g_backend_ops->operator_var(res, this);
    return res;
}

Matrix *Matrix::sigmoid() {
    Matrix *res = allocTmpMatrix(this);
    g_backend_ops->operator_sigmoid(res);
    return res;
}

Matrix *Matrix::sigmoid_prime() {
    Matrix *res = allocTmpMatrix(this);
    g_backend_ops->operator_sigmoid_prime(res);
    return res;
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
    g_backend_ops->operator_init_weight(this, sigma, mean);
}

void Matrix::init_weight_uniform(DATATYPE sigma) {
    g_backend_ops->operator_init_weight_uniform(this, sigma);
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
    assert(cpu_ver > gpu_ver);
    assert(allocated && initialized);
    commited = true;
    g_gpu_backend_ops->cp_to_device(data_device, data, shape.size()*sizeof(DATATYPE));
    gpu_ver = cpu_ver;
}

void Matrix::cp_from_device() {
    assert(cpu_ver < gpu_ver);
    g_gpu_backend_ops->cp_from_device(data, data_device, shape.size()*sizeof(DATATYPE));
    cpu_ver = gpu_ver;
}

void Matrix::sync() {
    if (cpu_ver < gpu_ver) {
        cp_from_device();
    } else if (cpu_ver > gpu_ver) {
        cp_to_device();
    }
}

bool Matrix::is_sync() const {
    return cpu_ver == gpu_ver;
}

void Matrix::increase_cpu_ver() {
    cpu_ver++;
}

void Matrix::increase_gpu_ver() {
    gpu_ver++;
}

TrainingData::TrainingData(int input_layer_size, int _y)
    : y(_y) {  
    x.reserve(input_layer_size);
}

TrainingData::~TrainingData() {
}