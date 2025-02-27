#include "matrix.h"

#include <iostream>
#include <assert.h>
#include <string.h>
#include <vector>

Matrix::Matrix(Shape _shape)
        : initialized(false),
        allocated(false),
        shape(_shape) {
    data = new DATATYPE[shape.size()];
    allocated = true;
    zero();
}

Matrix::Matrix(const Matrix &m):
    initialized(m.initialized),
    allocated(false),
    shape(m.shape) {
    assert(initialized);
    data = new DATATYPE[shape.size()];
    allocated = true;
    memcpy(data, m.data, sizeof(DATATYPE) * shape.rowCnt * shape.colCnt);
}

Matrix::~Matrix() {
    assert(initialized && allocated);
    delete [] data;
    data = nullptr;
}

Matrix *Matrix::zero() {
    assert(allocated);
    memset(data, 0, sizeof(DATATYPE) * shape.size());
    initialized = true;
    return this;
}

void Matrix::checkShape(const Matrix &m) {
    if (!(this->getShape() == m.getShape())) {
        std::cerr << 
            "matrix shape missmatch." << 
            this->getShape() << " vs " << m.getShape()<< 
            std::endl;
    }
    if (!m.initialized) {
        std::cerr << "matrix not initialized..." << std::endl;
    }
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

Matrix *Matrix::operator+(const Matrix &m) {
    checkShape(m);
    Matrix *res = allocTmpMatrix(this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*res)[i][j] += m[i][j];
        }
    }
    return res;
}

Matrix *Matrix::operator+(int dt) {
    Matrix *res = allocTmpMatrix(this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            
            (*res)[i][j] += dt;
        }
    }
    return res;
}

Matrix *Matrix::operator-(int dt) {
    Matrix *res = allocTmpMatrix(this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*res)[i][j] -= dt;
        }
    }
    return res;
}

Matrix *Matrix::operator-() {
    Matrix *res = allocTmpMatrix(this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            auto &r = (*res)[i][j];
            r = -r;
        }
    }
    return res;
}

Matrix *operator-(int v, const Matrix &m) {
    Matrix *res = allocTmpMatrix(m);
    for (uint i = 0; i < m.shape.rowCnt; ++i) {
        for (uint j = 0; j < m.shape.colCnt; ++j) {
            auto &r = (*res)[i][j];
            r = v-r;
        }
    }
    return res;
}

Matrix *Matrix::operator-(const Matrix &m) {
    checkShape(m);
    Matrix *res = allocTmpMatrix(this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*res)[i][j] -= m[i][j];
        }
    }
    return res;
}

Matrix *Matrix::operator*(const Matrix &m) {
    checkShape(m);
    Matrix *res = allocTmpMatrix(this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*res)[i][j] *= m[i][j];
        }
    }
    return res;
}

Matrix *Matrix::operator*(DATATYPE v) {
    Matrix *res = allocTmpMatrix(this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*res)[i][j] *= v;
        }
    }
    return res;
}

Matrix *Matrix::operator/(DATATYPE v) {
    Matrix *res = allocTmpMatrix(this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*res)[i][j] /= v;
        }
    }
    return res;
}

Matrix& Matrix::operator=(const Matrix &m) {
    assert(m.initialized);
    this->reShape(m.shape);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*this)[i][j] = m[i][j];
        }
    }
    return *this;
}

DATATYPE *Matrix::operator[](unsigned int index) const {
    assert(index < shape.rowCnt);
    return (DATATYPE *)&(data[index*shape.colCnt]);
}

Shape Matrix::getShape() const {
    return shape;
}

Matrix *Matrix::dot(const Matrix &m) {
    assert(m.shape.rowCnt == shape.colCnt);
    Matrix *res = allocTmpMatrix(Shape(shape.rowCnt, m.shape.colCnt));

    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint k = 0; k < shape.colCnt; ++k) {
            for (uint j = 0; j < m.shape.colCnt; ++j) {
                (*res)[i][j] += (*this)[i][k] * m[k][j];
            }
        }
    }
    return res;
}

Matrix *Matrix::transpose() {
    Matrix *res = allocTmpMatrix(Shape(shape.colCnt, shape.rowCnt));
    for (uint i = 0; i < shape.colCnt; ++ i) {
        for (uint j = 0; j < shape.rowCnt; ++ j) {
            (*res)[i][j] = (*this)[j][i];
        }
    }
    return res;
}

bool Matrix::valid(uint x, uint y) const {
    return allocated && initialized && x < shape.rowCnt && y < shape.colCnt;
}

void Matrix::reShape(Shape _shape) {
    assert(allocated && initialized);
    delete []data;
    shape = _shape;
    data = new DATATYPE[shape.size()];
    zero();
}

Matrix *Matrix::assign(Matrix *other) {
    assert(allocated && initialized);
    checkShape(other->getShape());
    memcpy(data, other->data, sizeof(DATATYPE) * shape.size());
    return this;
}

DATATYPE sigmoid_double(DATATYPE z) {
    return 1./(1.+exp(-z));
}

Matrix *sigmoid(const Matrix &m) {
    Shape shape = m.getShape();
    Matrix *res = allocTmpMatrix(m);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*res)[i][j] = sigmoid_double((*res)[i][j]);
        }
    }
    return res;
}

Matrix *sigmoid_prime(const Matrix &m) {
    return *sigmoid(m) * *(1 - *sigmoid(m));
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

void freeTmpMatrix() {
    for (auto p : tmpMatrics) {
        delete p;
    }
    tmpMatrics.clear();
}