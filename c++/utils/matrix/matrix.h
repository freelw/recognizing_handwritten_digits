#ifndef MATRIX_H
#define MATRIX_H

#include <ostream>
#include <vector>

using namespace std;

struct Shape {
    int rowCnt;
    int colCnt;
    Shape(int r, int c): rowCnt(r), colCnt(c) {}
    bool operator==(const Shape &s) {
        return rowCnt == s.rowCnt && colCnt == s.colCnt;
    }
    friend ostream &operator<<(ostream &output, const Shape &s) {
        output << "(" << s.rowCnt << ", " << s.colCnt << ")";
        return output;
    }
};

class Matrix {

public:
    Matrix(Shape _shape)
        : initialized(false),
        shape(_shape) {}

    Matrix(const Matrix &m);
    Matrix& zero();
    friend ostream &operator<<(ostream &output, const Matrix &m);
    Matrix operator+(const Matrix &m) const;
    Matrix operator+(int dt) const;
    Matrix operator-(int dt) const;
    Matrix operator-() const;
    Matrix operator-(const Matrix &m) const;
    Matrix operator*(const Matrix &m) const;
    Matrix operator*(double) const;
    Matrix operator/(double) const;
    Matrix& operator=(const Matrix &m);
    friend Matrix operator-(int, const Matrix &m);
    std::vector<double>& operator[](unsigned int index);
    Matrix& setAll(double v);
    Shape getShape() const;
    Matrix dot(Matrix &m);
    Matrix transpose();
private:
    void checkShape(const Matrix &m) const;
private:
    bool initialized;
    Shape shape;
    std::vector<std::vector<double>> data;
};

class TrainingData {
public:
    TrainingData(int, int);
    Matrix x;
    int y;
};

#endif