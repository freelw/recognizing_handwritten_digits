#include "autograd/parameters.h"

namespace autograd {

    Parameters::Parameters(Node *node) {
        w = node;
        m = allocTmpMatrix(w->getShape());
        v = allocTmpMatrix(w->getShape());
        t = 0;
    }

    Parameters::~Parameters() {
    }

    void Parameters::zero_grad() {
        w->zero_grad();
    }

    Matrix *Parameters::get_weight() {
        return w->get_weight();
    }

    Matrix *Parameters::get_grad() {
        return w->get_grad();
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
        t ++;
    }

} // namespace autograd