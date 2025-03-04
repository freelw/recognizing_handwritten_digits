#include "layers/layers.h"
#include <iostream>

void testgrad() {


    Rnn rnn(3, 4, 0.1);
    RnnContext *ctx = rnn.init();


    std::vector<Matrix *> inputs;
    for (int i = 0; i < 3; ++ i) {
        inputs.push_back(new Matrix(Shape(3, 1)));
    }

    (*(inputs[0]))[0][0] = 1;
    (*(inputs[1]))[1][0] = 2;
    (*(inputs[2]))[2][0] = 3;

    RnnRes res = rnn.forward(ctx, inputs, nullptr);

    for (int i = 0; i < 3; ++ i) {
        std::cout << *res.states[i] << std::endl;
    }

    for (int i = 0; i < 3; ++ i) {
        std::cout << *inputs[i] << std::endl;
    }
    rnn.release(ctx);

    for (int i = 0; i < 3; ++ i) {
        delete inputs[i];
    }
    freeTmpMatrix();
}