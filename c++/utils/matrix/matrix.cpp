#include "matrix.h"

#include <iostream>
#include <assert.h>
#include <string.h>

Matrix::Matrix(Shape _shape)
        : initialized(false),
        allocated(false),
        shape(_shape) {
    data = new DATATYPE[shape.size()];
    // cout <<  "allocate " << shape.size() << " double" << endl;
    allocated = true;
    zero();
}

Matrix::Matrix(const Matrix &m):
    initialized(m.initialized),
    allocated(false),
    shape(m.shape) {
    assert(initialized);
    data = new DATATYPE[shape.size()];
    // cout <<  "allocate " << shape.size() << " double" << endl;
    allocated = true;
    memcpy(data, m.data, sizeof(DATATYPE) * shape.rowCnt * shape.colCnt);
}

Matrix::~Matrix() {
    assert(initialized && allocated);
    delete [] data;
    data = nullptr;
}

Matrix& Matrix::zero() {
    assert(allocated);
    // cout << "sizeof(double) : " << sizeof(double) << endl;
    // cout << "shape.size() : " << shape.size() << endl;
    memset(data, 0, sizeof(DATATYPE) * shape.size());
    initialized = true;
    return *this;
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

Matrix Matrix::operator+(const Matrix &m) {
    checkShape(m);
    Matrix res(*this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            // assert(res.valid(i, j) && m.valid(i, j));
            res[i][j] += m[i][j];
        }
    }
    return res;
}

Matrix Matrix::operator+(int dt) {
    Matrix res(*this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            // assert(res.valid(i, j));
            res[i][j] += dt;
        }
    }
    return res;
}

Matrix Matrix::operator-(int dt) {
    Matrix res(*this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            // assert(res.valid(i, j));
            res[i][j] -= dt;
        }
    }
    return res;
}

Matrix Matrix::operator-() {
    Matrix res(*this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            // assert(res.valid(i, j));
            res[i][j] = -res[i][j];
        }
    }
    return res;
}

Matrix operator-(int v, const Matrix &m) {
    Matrix res(m);
    for (uint i = 0; i < m.shape.rowCnt; ++i) {
        for (uint j = 0; j < m.shape.colCnt; ++j) {
            // assert(res.valid(i, j));
            res[i][j] = v-res[i][j];
        }
    }
    return res;
}

Matrix Matrix::operator-(const Matrix &m) {
    checkShape(m);
    Matrix res(*this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            // assert(res.valid(i, j) && m.valid(i, j));
            res[i][j] -= m[i][j];
        }
    }
    return res;
}

Matrix Matrix::operator*(const Matrix &m) {
    checkShape(m);
    Matrix res(*this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            // assert(res.valid(i, j) && m.valid(i, j));
            res[i][j] *= m[i][j];
        }
    }
    return res;
}

Matrix Matrix::operator*(double v) {
    Matrix res(*this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            // assert(res.valid(i, j));
            res[i][j] *= v;
        }
    }
    return res;
}

Matrix Matrix::operator/(double v) {
    Matrix res(*this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            // assert(res.valid(i, j));
            res[i][j] /= v;
        }
    }
    return res;
}

Matrix& Matrix::operator=(const Matrix &m) {
    assert(m.initialized);
    this->reShape(m.shape);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            // assert(this->valid(i, j) && m.valid(i, j));
            (*this)[i][j] = m[i][j];
        }
    }
    return *this;
}

DATATYPE *Matrix::operator[](unsigned int index) const {
    assert(index < shape.rowCnt);
    // cout << "data : " << data << endl;
    // cout << "index*shape.colCnt : " << index*shape.colCnt << endl;
    // cout << "index : " << index << endl;
    // cout << "shape : " << shape << endl;
    // cout << "&(data[index*shape.colCnt]) : " << &(data[index*shape.colCnt]) << endl;
    return (DATATYPE *)&(data[index*shape.colCnt]);
}

Shape Matrix::getShape() const {
    return shape;
}

Matrix Matrix::dot(const Matrix &m) {
    assert(m.shape.rowCnt == shape.colCnt);
    Matrix res(Shape(shape.rowCnt, m.shape.colCnt));
    res.zero();

    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint k = 0; k < shape.colCnt; ++k) {
            for (uint j = 0; j < m.shape.colCnt; ++j) {
                // assert(res.valid(i, j) && this->valid(i, k) && m.valid(k, j));
                // if (!(res.valid(i, j) && this->valid(i, k) && m.valid(k, j))) {
                //     cout << res.valid(i, j) << " " <<  this->valid(i, k) << " " <<  m.valid(k, j) << endl;
                //     cout << "bug" << endl;
                // }
                res[i][j] += (*this)[i][k] * m[k][j];
            }
        }
    }
    return res;
}

Matrix Matrix::transpose() {
    Matrix res(Shape(shape.colCnt, shape.rowCnt));
    res.zero();
    for (uint i = 0; i < shape.colCnt; ++ i) {
        for (uint j = 0; j < shape.rowCnt; ++ j) {
            // assert(res.valid(i, j) && this->valid(j, i));
            res[i][j] = (*this)[j][i];
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

DATATYPE sigmoid_double(DATATYPE z) {
    return 1./(1.+exp(-z));
}

Matrix sigmoid(const Matrix &m) {
    Shape shape = m.getShape();
    Matrix res(m);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            // assert(res.valid(i, j));
            res[i][j] = sigmoid_double(res[i][j]);
        }
    }
    return res;
}

Matrix sigmoid_prime(const Matrix &m) {
    return sigmoid(m) * (1 - sigmoid(m));
}