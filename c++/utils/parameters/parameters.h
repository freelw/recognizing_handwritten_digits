#ifndef PARAMETERS_H
#define PARAMETERS_H
#include "matrix/matrix.h"
#include <assert.h>

class Parameters {
    public:
        Parameters(Shape shape);
        ~Parameters();
        void set_grad(Matrix * _grad);
        void inc_grad(Matrix * _grad);
        void zero_grad();
        Matrix *get_weight();
        Matrix *get_grad();
        Matrix *get_m();
        Matrix *get_v();
        int get_t();
        void inc_t();
        friend std::ostream & operator<<(std::ostream &output, const Parameters &p);
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