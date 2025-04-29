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

typedef float DATATYPE;

class Matrix {

public:
    Matrix(Shape _shape);
    Matrix(const Matrix &m);
    Matrix(const std::vector<DATATYPE> &v);
    ~Matrix();
    Matrix *zero();
    friend ostream &operator<<(ostream &output, const Matrix &m);
    Matrix *expand_add(Matrix &m);
    Matrix *operator+(Matrix &m);
    Matrix *operator+=(Matrix &m);
    Matrix *operator+(DATATYPE dt);
    Matrix *operator-(DATATYPE dt);
    Matrix *operator-();
    Matrix *operator-(const Matrix &m);
    Matrix *operator-=(const Matrix &m);
    Matrix *operator*(const Matrix &m);
    Matrix *operator*(DATATYPE);
    Matrix *operator*=(DATATYPE);
    Matrix *operator/(DATATYPE);
    Matrix *Relu();
    Matrix *Relu_prime();
    Matrix *tanh();
    Matrix *tanh_prime();
    Matrix *sigmoid();
    Matrix *sigmoid_prime();
    Matrix& operator=(const Matrix &m);
    Matrix *pow2();
    Shape getShape() const;
    Matrix *at(Matrix &m);
    Matrix *transpose();
    bool valid(uint x, uint y) const;
    void reShape(Shape shape);
    Matrix *assign(Matrix *);
    bool checkShape(const Matrix &m);
    Matrix *sum(uint dim);
    std::vector<Matrix *> split(uint dim);
    Matrix *fill(DATATYPE value);
    std::vector<uint> argMax();
    std::vector<DATATYPE> avg();
    std::vector<DATATYPE> var();
    friend Matrix *operator-(DATATYPE, const Matrix &m);
    void init_weight(DATATYPE sigma, DATATYPE mean = 0);
    void init_weight_uniform(DATATYPE sigma);
    void set_val(int i, int j, DATATYPE val);
    DATATYPE get_val(int i, int j) const;
    
    void sync();
    bool is_sync() const;
    void increase_cpu_ver();
    void increase_gpu_ver();
    DATATYPE *getLowLevelData() const;
    void *getLowLevelDataDevice() const;
    void cp_to_device();
    void cp_from_device();
    
private:
    DATATYPE* operator[](unsigned int index) const;
    

private:
    bool initialized;
    bool allocated;
    Shape shape;
    DATATYPE *data;
    void *data_device;
    bool commited;
    int cpu_ver;
    int gpu_ver;

friend class CPUBackendOps;
friend class GPUBackendOps;
};

Matrix *sigmoid(const Matrix &m);
Matrix *sigmoid_prime(const Matrix &m);
Matrix *allocTmpMatrix(Matrix *m);
Matrix *allocTmpMatrix(const Matrix &m);
Matrix *allocTmpMatrix(const Shape & shape);
Matrix *allocTmpMatrix(const std::vector<DATATYPE> &v);
void freeTmpMatrix();

class TrainingData {
public:
    TrainingData(int, int);
    ~TrainingData();
    std::vector<DATATYPE> x;
    uint y;
};

#endif