#include "layers/layers.h"
#include "optimizers/optimizers.h"
#include "rnnlm.h"
#include <iostream>

void testgrad() {
    std::vector<Matrix *> inputs;
    for (int i = 0; i < 3; ++ i) {
        inputs.push_back(new Matrix(Shape(3, 1)));
    }

    (*(inputs[0]))[0][0] = 1;
    (*(inputs[1]))[1][0] = 1;
    (*(inputs[2]))[2][0] = 1;

    Rnn *rnn = new Rnn(3, 4, 0.1, false);
    RnnLM lm(rnn, 3, false);
    RnnLMContext *ctx = lm.init();
    Adam adam(lm.get_parameters(), 0.001);
    Matrix *res = lm.forward(ctx, inputs);
    std::cout << "res : " << *res << std::endl;
    CrossEntropyLoss loss_fn({2, 1, 2});
    CrossEntropyLossContext *ce_ctx = (CrossEntropyLossContext *)loss_fn.init();
    auto loss = loss_fn.forward(ce_ctx, res);
    std::cout << "loss : " << *loss << std::endl;
    auto grad = loss_fn.backward(ce_ctx, nullptr);
    loss_fn.release(ce_ctx);
    std::cout << "grad : " << *grad << std::endl;
    lm.backward(ctx, grad);
    lm.clip_grad(1);
    adam.step();

    // print all parameters
    auto parameters = lm.get_parameters();
    for (auto p : parameters) {
        std::cout << *p << std::endl;
    }
    
    lm.release(ctx);
    for (int i = 0; i < 3; ++ i) {
        delete inputs[i];
    }
    delete rnn;
    freeTmpMatrix();
}