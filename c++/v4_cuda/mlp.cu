#include "mlp.cuh"

Matrix *allocMatrix(Shape shape) {
    return new Matrix(shape);
}

namespace autograd_cuda {


MLP::MLP(uint _input, const std::vector<uint> &_outputs) {

    mW1 = allocMatrix(Shape(_outputs[0], _input));
    mb1 = allocMatrix(Shape(_outputs[0], 1));
    mW2 = allocMatrix(Shape(_outputs[1], _outputs[0]));
    mb2 = allocMatrix(Shape(_outputs[1], 1));

    auto sigma = 0.02;
    mW1->init_weight(sigma);
    mb1->init_weight(sigma);
    mW2->init_weight(sigma);
    mb2->init_weight(sigma);
    
    W1 = new Node(mW1, true);
    b1 = new Node(mb1, true);
    W2 = new Node(mW2, true);
    b2 = new Node(mb2, true);

    W1->require_grad();
    b1->require_grad();
    W2->require_grad();
    b2->require_grad();

    PW1 = new Parameters(W1);
    Pb1 = new Parameters(b1);
    PW2 = new Parameters(W2);
    Pb2 = new Parameters(b2);
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

Node *MLP::forward(Node *input) {
    auto Z1 = W1->at(input)->expand_add(b1)->Relu();
    auto Z2 = W2->at(Z1)->expand_add(b2);
    return Z2;
}

std::vector<Parameters*> MLP::get_parameters() {
    std::vector<Parameters*> res;
    res.push_back(PW1);
    res.push_back(Pb1);
    res.push_back(PW2);
    res.push_back(Pb2);
    return res;
}

}// namespace autograd