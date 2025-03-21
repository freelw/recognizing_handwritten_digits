#include "matrix/matrix.h"

int main() {
  Matrix *w = allocTmpMatrix(Shape(256, 256));
  Matrix *x = allocTmpMatrix(Shape(256, 256));


    for (int i = 0; i < 1000; i++) {
        // w = w * x
        x = w->at(*x);
    }

    freeTmpMatrix();
    return 0;
}