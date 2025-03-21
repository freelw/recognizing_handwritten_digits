#include "matrix/matrix.h"

int main() {
    Matrix *w = allocTmpMatrix(Shape(256, 256));
    Matrix *x = allocTmpMatrix(Shape(256, 256));
    init_weight(w, 0.01);
    init_weight(x, 0.01);
    for (int i = 0; i < 10000; i++) {
        // w = w * x
        x = w->at(*x);
    }
    freeTmpMatrix();
    return 0;
}