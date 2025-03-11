#include "mlp.h"

Matrix *allocMatrix(Shape shape) {
    return new Matrix(shape);
}

namespace autograd {


MLP::MLP(uint _input, const std::vector<uint> &_outputs) {

    mW1 = allocMatrix(Shape(_outputs[0], _input));
    mb1 = allocMatrix(Shape(_outputs[0], 1));
    mW2 = allocMatrix(Shape(_outputs[1], _outputs[0]));
    mb2 = allocMatrix(Shape(_outputs[1], 1));

    init_weight(mW1, 0.1);
    init_weight(mb1, 0.1);
    init_weight(mW2, 0.1);
    init_weight(mb2, 0.1);
    
    W1 = allocNode(mW1);
    b1 = allocNode(mb1);
    W2 = allocNode(mW2);
    b2 = allocNode(mb2);

    W1->require_grad();
    b1->require_grad();
    W2->require_grad();
    b2->require_grad();

    PW1 = new Parameters(W1);
    Pb1 = new Parameters(b1);
    PW2 = new Parameters(W2);
    Pb2 = new Parameters(b2);


    // W1 = new Parameters(allocNode(mW1));
    // b1 = new Parameters(allocNode(mb1));
    // W2 = new Parameters(allocNode(mW2));
    // b2 = new Parameters(allocNode(mb2));

    // W1 = new Parameters(allocMatrix(Shape(_outputs[0], _input)));
    // b1 = new Parameters(allocMatrix(Shape(_outputs[0], 1)));
    // W2 = new Parameters(allocMatrix(Shape(_outputs[1], _outputs[0])));
    // b2 = new Parameters(allocMatrix(Shape(_outputs[1], 1)));
}

MLP::~MLP() {
    delete mW1;
    delete mb1;
    delete mW2;
    delete mb2;
    
    delete W1;
    delete b1;
    delete W2;
    delete b2;
    
    delete PW1;
    delete Pb1;
    delete PW2;
    delete Pb2;
}

Matrix *MLP::forward(Matrix *input) {
    auto Z1 = W1->at(input)->expand_add(b1->get_weight())->Relu();
    auto Z2 = W2->get_weight()->at(Z1)->expand_add(b2->get_weight());
    return Z2;
}

}// namespace autograd