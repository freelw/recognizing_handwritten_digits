
#include "models.h"
#include <iostream>

void testgrad_bak() {
    MLP m(2, {3, 3}, false);
    m.init();

    auto parameters =  m.get_parameters();
    (*parameters[0]->get_weight())[0][0] = 0.9;
    (*parameters[0]->get_weight())[1][0] = -0.9;
    (*parameters[2]->get_weight())[0][0] = 0.9;
    (*parameters[2]->get_weight())[1][0] = -0.9;
    cout << *parameters[0]->get_weight() << endl;
    cout << *parameters[2]->get_weight() << endl;

    Matrix *input = allocTmpMatrix(Shape(2, 1));
    (*input)[0][0] = 10;
    (*input)[1][0] = 11;
    std::vector<uint> labels;
    labels.push_back(1);

    m.backward(input, labels);
    for (auto p : m.get_parameters()) {
        cout << *p << endl;
    }
    freeTmpMatrix();
}

void optimize(const std::vector<Parameters*> &parameters, DATATYPE lr, int epoch);

void testgrad() {

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
    
    for (uint k = 0; k < 20; ++ k) {
        m.zero_grad();
        auto loss = m.backward(input, labels);
        cout << k << " loss : " << loss << endl;
        optimize(m.get_parameters(), 0.001, k);
    }
    for (auto p : m.get_parameters()) {
        cout << *p << endl;
    }
    freeTmpMatrix();
}