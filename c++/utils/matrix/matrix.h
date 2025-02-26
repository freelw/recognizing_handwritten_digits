#ifndef MATRIX_H
#define MATRIX_H

#include <ostream>

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
    Matrix& zero();
    friend ostream &operator<<(ostream &output, const Matrix &m);
    Matrix operator+(const Matrix &m);
    Matrix operator+(int dt);
    Matrix operator-(int dt);
    Matrix operator-();
    Matrix operator-(const Matrix &m);
    Matrix operator*(const Matrix &m);
    Matrix operator*(double);
    Matrix operator/(double);
    Matrix& operator=(const Matrix &m);
    friend Matrix operator-(int, const Matrix &m);
    DATATYPE* operator[](unsigned int index) const;
    // Matrix& setAll(double v);
    Shape getShape() const;
    Matrix dot(const Matrix &m);
    Matrix transpose();
    bool valid(uint x, uint y) const;
    void reShape(Shape shape);
private:
    void checkShape(const Matrix &m);
private:
    bool initialized;
    bool allocated;
    Shape shape;
    DATATYPE *data;
};

class TrainingData {
public:
    TrainingData(int, int);
    Matrix x;
    int y;
};

#endif