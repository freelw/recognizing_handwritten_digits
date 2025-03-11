#include <iostream>
#include "autograd/node.h"
#include "autograd/optimizers.h"

/*void testgrad() {

    MLP m(7, {4, 3}, false);
    m.init();

    auto parameters =  m.get_parameters();

    (*parameters[0]->get_weight())[0][0] = 0.9;
    (*parameters[0]->get_weight())[1][0] = -0.9;
    (*parameters[2]->get_weight())[0][0] = 0.9;
    (*parameters[2]->get_weight())[1][0] = -0.9;
    // cout << *parameters[0]->get_weight() << endl;
    // cout << *parameters[2]->get_weight() << endl;

    Matrix *input = allocTmpMatrix(Shape(7, 30));
    std::vector<uint> labels;
    for (uint j = 0; j < 15; ++ j) {
        for (uint i = 0; i < 7; ++ i) {
            (*input)[i][j*2] = 10 + i;
            (*input)[i][j*2+1] = 10 - i;
        }
        labels.push_back(1);
        labels.push_back(0);
    }

    Adam adam(m.get_parameters(), 0.001);
    
    for (uint k = 0; k < 20; ++ k) {
        adam.zero_grad();
        auto loss = m.backward(input, labels);
        cout << k << " loss : " << loss << endl;
        adam.step();
    }
    for (auto p : m.get_parameters()) {
        cout << *p << endl;
    }
    freeTmpMatrix();
}*/

void testgrad() {
    Matrix *mW1 = allocTmpMatrix(Shape(4, 7));
    Matrix *mb1 = allocTmpMatrix(Shape(4, 1));
    Matrix *mW2 = allocTmpMatrix(Shape(3, 4));
    Matrix *mb2 = allocTmpMatrix(Shape(3, 1));
    Matrix *mX = allocTmpMatrix(Shape(7, 30));

    (*mW1)[0][0] = 0.9;
    (*mW1)[1][0] = -0.9;
    (*mW2)[0][0] = 0.9;
    (*mW2)[1][0] = -0.9;

    std::cout << *mW1 << std::endl;
    std::cout << *mW2 << std::endl;

    std::vector<uint> labels;
    for (uint j = 0; j < 15; ++ j) {
        for (uint i = 0; i < 7; ++ i) {
            (*mX)[i][j*2] = 10 + i;
            (*mX)[i][j*2+1] = 10 - i;
        }
        labels.push_back(1);
        labels.push_back(0);
    }

    std::vector<autograd::Parameters *> parameters;
    auto W1 = autograd::allocNode(mW1);
    auto b1 = autograd::allocNode(mb1);
    auto W2 = autograd::allocNode(mW2);
    auto b2 = autograd::allocNode(mb2);
    auto X = autograd::allocNode(mX);
    parameters.push_back(new autograd::Parameters(W1));
    parameters.push_back(new autograd::Parameters(b1));
    parameters.push_back(new autograd::Parameters(W2));
    parameters.push_back(new autograd::Parameters(b2));

    autograd::Adam adam(parameters, 0.001);
    auto Z1 = X->at(W1)->expand_add(b1)->Relu();
    auto Z2 = Z1->at(W2)->expand_add(b2)->Relu();
    auto loss = Z2->CrossEntropy(labels);

    adam.zero_grad();
    loss->backward();
    adam.step();

    std::cout << W1->get_weight() << std::endl;
    std::cout << W1->get_grad() << std::endl;
    std::cout << b1->get_weight() << std::endl;
    std::cout << b1->get_grad() << std::endl;
    std::cout << W2->get_weight() << std::endl;
    std::cout << W2->get_grad() << std::endl;
    std::cout << b2->get_weight() << std::endl;
    std::cout << b2->get_grad() << std::endl;
    autograd::freeAllNodes();
    autograd::freeAllEdges();
    freeTmpMatrix();

    for (auto p : parameters) {
        delete p;
    }
}

int main(int argc, char *argv[]) {
    testgrad();
    return 0;
}