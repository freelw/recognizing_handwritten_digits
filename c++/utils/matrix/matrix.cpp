#include "matrix.h"

#include <iostream>
#include <assert.h>

Matrix::Matrix(const Matrix &m):
    shape(m.shape),
    initialized(m.initialized),
    data(m.data) {
}

Matrix& Matrix::zero() {
    return setAll(0);
}

void Matrix::checkShape(const Matrix &m) const {
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
    for (auto i = 0; i < m.shape.rowCnt; ++ i) {
        if (i > 0) {
            output << " ";
        }
        output << "[";
        for (auto j = 0; j < m.shape.colCnt-1; ++ j) {
            output << m.data[i][j] << ", ";
        }
        output << m.data[i][m.shape.colCnt-1] << "]";
        if (i < m.shape.rowCnt-1) {
            output << endl;
        }
    }
    output << "]" << endl;
    return output;
}

Matrix Matrix::operator+(const Matrix &m) const{
    checkShape(m);
    Matrix res(*this);
    for (auto i = 0; i < shape.rowCnt; ++i) {
        for (auto j = 0; j < shape.colCnt; ++j) {
            res.data[i][j] += m.data[i][j];
        }
    }
    return res;
}

Matrix Matrix::operator+(int dt) const{
    Matrix res(*this);
    for (auto i = 0; i < shape.rowCnt; ++i) {
        for (auto j = 0; j < shape.colCnt; ++j) {
            res.data[i][j] += dt;
        }
    }
    return res;
}

Matrix Matrix::operator-(int dt) const{
    Matrix res(*this);
    for (auto i = 0; i < shape.rowCnt; ++i) {
        for (auto j = 0; j < shape.colCnt; ++j) {
            res.data[i][j] -= dt;
        }
    }
    return res;
}

Matrix Matrix::operator-() const{
    Matrix res(*this);
    for (auto i = 0; i < shape.rowCnt; ++i) {
        for (auto j = 0; j < shape.colCnt; ++j) {
            res.data[i][j] = -res.data[i][j];
        }
    }
    return res;
}

Matrix operator-(int v, const Matrix &m) {
    Matrix res(m);
    for (auto i = 0; i < m.shape.rowCnt; ++i) {
        for (auto j = 0; j < m.shape.colCnt; ++j) {
            res.data[i][j] = v-res.data[i][j];
        }
    }
    return res;
}

Matrix Matrix::operator-(const Matrix &m) const {
    checkShape(m);
    Matrix res(*this);
    for (auto i = 0; i < shape.rowCnt; ++i) {
        for (auto j = 0; j < shape.colCnt; ++j) {
            res.data[i][j] -= m.data[i][j];
        }
    }
    return res;
}

Matrix Matrix::operator*(const Matrix &m) const {
    checkShape(m);
    Matrix res(*this);
    for (auto i = 0; i < shape.rowCnt; ++i) {
        for (auto j = 0; j < shape.colCnt; ++j) {
            res.data[i][j] *= m.data[i][j];
        }
    }
    return res;
}

Matrix Matrix::operator*(double v) const {
    Matrix res(*this);
    for (auto i = 0; i < shape.rowCnt; ++i) {
        for (auto j = 0; j < shape.colCnt; ++j) {
            res.data[i][j] *= v;
        }
    }
    return res;
}

Matrix Matrix::operator/(double v) const {
    Matrix res(*this);
    for (auto i = 0; i < shape.rowCnt; ++i) {
        for (auto j = 0; j < shape.colCnt; ++j) {
            res.data[i][j] /= v;
        }
    }
    return res;
}

Matrix& Matrix::operator=(const Matrix &m) {
    assert(m.initialized);
    shape = m.shape;
    this->setAll(0);
     for (auto i = 0; i < shape.rowCnt; ++i) {
        for (auto j = 0; j < shape.colCnt; ++j) {
            data[i][j] = m.data[i][j];
        }
    }
    return *this;
}

std::vector<double>& Matrix::operator[](unsigned int index) {
    return data[index];
}

Matrix& Matrix::setAll(double v) {
    data.clear();
    for (auto i = 0; i < shape.rowCnt; ++i) {
        std::vector<double> tmp;
        for (auto j = 0; j < shape.colCnt; ++j) {
            tmp.emplace_back(v);
        }
        data.emplace_back(tmp);
    }
    initialized = true;
    return *this;
}

Shape Matrix::getShape() const {
    return shape;
}

Matrix Matrix::dot(Matrix &m) {
    Matrix res(Shape(shape.rowCnt, m.shape.colCnt));
    res.zero();
    for (auto i = 0; i < m.shape.colCnt; ++ i) {
        for (auto j = 0; j < shape.rowCnt; ++ j) {
            double tmp = 0;
            for (auto k = 0; k < shape.colCnt; ++ k) {
                tmp += m[k][i] * data[j][k];
            }
            res[j][i] = tmp;
        }
    }
    return res;
}

Matrix Matrix::transpose() {
    Matrix res(Shape(shape.colCnt, shape.rowCnt));
    res.zero();
    for (auto i = 0; i < shape.colCnt; ++ i) {
        for (auto j = 0; j < shape.rowCnt; ++ j) {
            res[i][j] = data[j][i]; 
        }
    }
    return res;
}
