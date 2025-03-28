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
    std::vector<uint> labels = {2, 3};
    autograd::Node *x = autograd::allocNode(input);
    x->require_grad();
    autograd::Node *y = layernorm.forward(x);

    auto loss = y->CrossEntropy(labels);
    cout << "y: " << endl;
    cout << *y->get_weight() << endl;
    cout << "loss: " << endl;
    cout << *loss->get_weight() << endl;

    loss->backward();

    cout << "gamma beta grad: " << endl;
    for (auto param : layernorm.parameters()) {
        cout << *param->get_grad() << endl;
    }

    cout << "x grad: " << endl;
    cout << *x->get_grad() << endl;

    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

int main() {
    test_layernorm();
    return 0;
}