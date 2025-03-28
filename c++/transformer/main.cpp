#include <iostream>
#include "layernorm.h"
using namespace std;

void test_layernorm() {
    auto normalized_shape = 6;
    LayerNorm layernorm(normalized_shape);
    Matrix *input = allocTmpMatrix(Shape(6, 2));
    for (int i = 0; i < 6; i++) {
        (*input)[i][0] = i;
        (*input)[i][1] = i+1;
    }
    autograd::Node *x = autograd::allocNode(input);
    autograd::Node *y = layernorm.forward(x);
    cout << "y: " << endl;
    cout << *y->get_weight() << endl;
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

int main() {
    test_layernorm();
    return 0;
}