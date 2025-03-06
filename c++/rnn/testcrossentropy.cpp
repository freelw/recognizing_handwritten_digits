#include "layers/layers.h"

#include <iostream>

void testcrossentropy() {
    std::vector<DATATYPE> input_data = {
        556.225, 1919.83, 2769.87, 2810.71, 2811.33, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34
    };
    Matrix *input = new Matrix(Shape(32, 1));
    for (uint i = 0; i < input_data.size(); i++) {
        (*input)[i][0] = input_data[i];
    }
    std::vector<uint> labels = {0};

    CrossEntropyLoss loss_fn(labels);
    CrossEntropyLossContext *ce_ctx = (CrossEntropyLossContext *)loss_fn.init();
    auto loss = loss_fn.forward(ce_ctx, input);
    std::cout << "loss : " << *loss << std::endl;

    Matrix *grad = loss_fn.backward(ce_ctx, nullptr);
    std::cout << "grad : " << *grad << std::endl;
    
    loss_fn.release(ce_ctx);
    delete input;
    freeTmpMatrix();
}