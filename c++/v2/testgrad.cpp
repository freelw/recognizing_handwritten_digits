
#include "models.h"
#include <iostream>

void testgrad() {

    MLP m(2, {3, 3}, false);
    m.init();


    Matrix *input = allocTmpMatrix(Shape(2, 1));
    (*input)[0][0] = 0;
    (*input)[1][0] = 1;
    std::vector<uint> labels;
    labels.push_back(1);

    m.backward(input, labels);

    for (auto & p : m.get_parameters()) {
        cout << *p << endl;
    }

    freeTmpMatrix();
}
