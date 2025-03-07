
#include "parameters.h"

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