
#include "models.h"
#include <iostream>

void testgrad() {

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
