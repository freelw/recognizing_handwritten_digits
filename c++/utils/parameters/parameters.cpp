
#include "parameters.h"
#include <string.h>
#include <iostream>

Parameters::Parameters(Shape shape) : grad(nullptr), t(0) {
    w = new Matrix(shape);
    m = new Matrix(shape);
    v = new Matrix(shape);
}
Parameters::~Parameters() {
    assert(w != nullptr);
    delete w;
    delete m;
    delete v;
}
void Parameters::set_grad(Matrix * _grad) {
    assert(grad == nullptr);
    grad = _grad;
}
void Parameters::inc_grad(Matrix * _grad) {
    if (grad == nullptr) {
        grad = _grad;
    } else {
        grad->checkShape(*_grad);
        *grad += *_grad;
    }
}
void Parameters::zero_grad() {
    grad = nullptr;
}
Matrix *Parameters::get_weight() {
    return w;
}
Matrix *Parameters::get_grad() {
    return grad;
}
Matrix *Parameters::get_m() {
    return m;
}
Matrix *Parameters::get_v() {
    return v;
}

int Parameters::get_t() {
    return t;
}

void Parameters::inc_t() {
    t++;
}

std::string Parameters::serialize() {
    Shape shape = w->getShape();
    DATATYPE *w_data = w->getData();
    DATATYPE *m_data = m->getData();
    DATATYPE *v_data = v->getData();
    int data_width = sizeof(DATATYPE);
    int data_size = shape.size() * data_width;
    

    int tot_size = 0;
    tot_size += sizeof(data_width);
    tot_size += sizeof(data_size);
    tot_size += sizeof(shape.rowCnt);
    tot_size += sizeof(shape.colCnt);
    tot_size += sizeof(t);
    tot_size += data_size; // w_data
    tot_size += data_size; // m_data
    tot_size += data_size; // v_data
    
    auto buffer = new char[tot_size];
    auto offset = 0;
    memcpy(buffer + offset, &data_width, sizeof(data_width));
    offset += sizeof(data_width);
    memcpy(buffer + offset, &data_size, sizeof(data_size));
    offset += sizeof(data_size);
    memcpy(buffer + offset, &shape.rowCnt, sizeof(shape.rowCnt));
    offset += sizeof(shape.rowCnt);
    memcpy(buffer + offset, &shape.colCnt, sizeof(shape.colCnt));
    offset += sizeof(shape.colCnt);
    memcpy(buffer + offset, &t, sizeof(t));
    offset += sizeof(t);
    memcpy(buffer + offset, w_data, data_size);
    offset += data_size;
    memcpy(buffer + offset, m_data, data_size);
    offset += data_size;
    memcpy(buffer + offset, v_data, data_size);
    std::string res((char *)buffer, tot_size);
    delete [] buffer;
    return res;
}

void Parameters::deserialize(char *buffer) {
    assert (w != nullptr);
    int data_width;
    int data_size;
    int rowCnt;
    int colCnt;
    auto offset = 0;
    memcpy(&data_width, buffer + offset, sizeof(data_width));
    if (data_width != sizeof(DATATYPE)) {
        std::cerr << "data width mismatch." << std::endl;
        abort();
    }
    offset += sizeof(data_width);
    memcpy(&data_size, buffer + offset, sizeof(data_size));
    offset += sizeof(data_size);
    memcpy(&rowCnt, buffer + offset, sizeof(rowCnt));
    offset += sizeof(rowCnt);
    memcpy(&colCnt, buffer + offset, sizeof(colCnt));
    offset += sizeof(colCnt);
    memcpy(&t, buffer + offset, sizeof(t));
    offset += sizeof(t);
    Shape shape(rowCnt, colCnt);
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wsign-compare"    
    assert(data_size == shape.size() * data_width);
    #pragma GCC diagnostic pop
    memcpy(w->getData(), buffer + offset, data_size);
    offset += data_size;
    memcpy(m->getData(), buffer + offset, data_size);
    offset += data_size;
    memcpy(v->getData(), buffer + offset, data_size);
}

std::ostream & operator<<(std::ostream &output, const Parameters &p) {
    output << std::endl << "weight : " << endl << "\t";
    Shape shape = p.w->getShape();
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        for (uint j = 0; j < shape.colCnt; ++ j) {
            output << (*p.w)[i][j] << " ";
        }
        output << endl << "\t";
    }
    output << std::endl << "grad : " << endl << "\t";
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        for (uint j = 0; j < shape.colCnt; ++ j) {
            output << (*p.grad)[i][j] << " ";
        }
        output << endl << "\t";
    }
    return output;
}