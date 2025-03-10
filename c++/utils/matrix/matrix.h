#ifndef MATRIX_H
#define MATRIX_H

#include <ostream>
#include <math.h>
#include <vector>

using namespace std;

struct Shape {
    uint rowCnt;
    uint colCnt;
    Shape(uint r, uint c): rowCnt(r), colCnt(c) {}
    bool operator==(const Shape &s) {
        return rowCnt == s.rowCnt && colCnt == s.colCnt;
    }
    friend ostream &operator<<(ostream &output, const Shape &s) {
        output << "(" << s.rowCnt << ", " << s.colCnt << ")";
        return output;
    }
    uint size() {
        return rowCnt * colCnt;
    }
};

// typedef double DATATYPE;
typedef float DATATYPE;

class Matrix {

public:
    Matrix(Shape _shape);
    Matrix(const Matrix &m);
    ~Matrix();
    Matrix *zero();
    friend ostream &operator<<(ostream &output, const Matrix &m);
    Matrix *expand_add(const Matrix &m);
    Matrix *operator+(const Matrix &m);
    Matrix *operator+=(const Matrix &m);
    Matrix *operator+(DATATYPE dt);
    Matrix *operator-(DATATYPE dt);
    Matrix *operator-();
    Matrix *operator-(const Matrix &m);
    Matrix *operator*(const Matrix &m);
    Matrix *operator*(DATATYPE);
    Matrix *operator*=(DATATYPE);
    Matrix *operator/(DATATYPE);
    Matrix *tanh();
    Matrix *tanh_prime();
    Matrix& operator=(const Matrix &m);
    Matrix *pow2();
    friend Matrix *operator-(DATATYPE, const Matrix &m);
    DATATYPE* operator[](unsigned int index) const;
    Shape getShape() const;
    Matrix *at(const Matrix &m);
    Matrix *transpose();
    bool valid(uint x, uint y) const;
    void reShape(Shape shape);
    Matrix *assign(Matrix *);
    void checkShape(const Matrix &m);
    Matrix *sum(uint dim);
    std::vector<Matrix *> split(uint dim);
    DATATYPE *getData() const;
private:
    bool initialized;
    bool allocated;
    Shape shape;
    DATATYPE *data;
};

class TrainingData {
public:
    TrainingData(int, int);
    ~TrainingData();
    Matrix *x;
    uint y;
};

Matrix *sigmoid(const Matrix &m);
Matrix *sigmoid_prime(const Matrix &m);
Matrix *allocTmpMatrix(Matrix *m);
Matrix *allocTmpMatrix(const Matrix &m);
Matrix *allocTmpMatrix(const Shape & shape);
void freeTmpMatrix();

#endif