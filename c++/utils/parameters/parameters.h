#ifndef PARAMETERS_H
#define PARAMETERS_H
#include "matrix/matrix.h"
#include <assert.h>

class Parameters {
    public:
        Parameters(Shape shape) : grad(nullptr), t(0) {
            w = new Matrix(shape);
            m = new Matrix(shape);
            v = new Matrix(shape);
        }
        ~Parameters() {
            assert(w != nullptr);
            delete w;
            delete m;
            delete v;
        }
        void set_grad(Matrix * _grad) {
            assert(grad == nullptr);
            grad = _grad;
        }
        void inc_grad(Matrix * _grad) {
            if (grad == nullptr) {
                grad = _grad;
            } else {
                grad->checkShape(*_grad);
                *grad += *_grad;
            }
        }
        void zero_grad() {
            grad = nullptr;
        }
        Matrix *get_weight() {
            return w;
        }
        Matrix *get_grad() {
            return grad;
        }
        Matrix *get_m() {
            return m;
        }
        Matrix *get_v() {
            return v;
        }
        int get_t() {
            return t;
        }
        void inc_t() {
            t++;
        }
        friend std::ostream & operator<<(std::ostream &output, const Parameters &p) {
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
    private:
        Parameters(const Parameters&);    
        Parameters& operator=(const Parameters&);
    private:
        Matrix *w;
        Matrix *grad;
        // m v for adam opt
        Matrix *m;
        Matrix *v;
        int t;
};



#endif